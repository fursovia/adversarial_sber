from copy import deepcopy
import random

import torch
from allennlp.models import Model
from allennlp.nn import util

from advsber.attackers.attacker import Attacker, AttackerOutput
from advsber.attackers.concat_sampling_fool import Position
from advsber.settings import TransactionsData, MASK_TOKEN
from advsber.dataset_readers.transactions_reader import TransactionsDatasetReader
from advsber.utils.data import data_to_tensors, decode_indexes, generate_transaction_amounts
from advsber.utils.metrics import word_error_rate_on_sequences


@Attacker.register("concat_fgsm")
class ConcatFGSM(Attacker):
    def __init__(
        self,
        classifier: Model,  # TransactionsClassifier
        reader: TransactionsDatasetReader,
        num_steps: int = 10,
        epsilon: float = 0.01,
        position: Position = Position.END,
        num_tokens_to_add: int = 2,
        total_amount: float = 5000,
        device: int = -1,
    ) -> None:
        super().__init__(classifier=classifier, reader=reader, device=device)
        self.classifier = self.classifier.train()
        self.num_steps = num_steps
        self.epsilon = epsilon

        self.emb_layer = util.find_embedding_layer(self.classifier).weight
        self.special_indexes = [
            self.classifier.vocab.get_token_index(token, "transactions") for token in self.SPECIAL_TOKENS
        ]

        self.position = position
        self.num_tokens_to_add = num_tokens_to_add
        self.total_amount = total_amount

    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:
        # get inputs to the model
        inputs = data_to_tensors(data_to_attack, reader=self.reader, vocab=self.classifier.vocab, device=self.device)

        # get original indexes of a sequence
        orig_indexes = inputs["transactions"]["tokens"]["tokens"]

        # original probability of the true label
        orig_prob = self.get_clf_probs(inputs)[self.label_to_index(data_to_attack.label)].item()

        adv_data = deepcopy(data_to_attack)
        amounts = generate_transaction_amounts(self.total_amount, self.num_tokens_to_add)
        if self.position == Position.END:
            adv_data.transactions = adv_data.transactions + [MASK_TOKEN] * self.num_tokens_to_add
            adv_data.amounts = adv_data.amounts + amounts
        else:
            raise NotImplementedError

        adv_inputs = data_to_tensors(adv_data, self.reader, self.classifier.vocab, self.device)

        # get mask and transaction embeddings
        emb_out = self.classifier.get_transaction_embeddings(transactions=adv_inputs["transactions"])

        # disable gradients using a trick
        embeddings = emb_out["transaction_embeddings"].detach()
        embeddings_splitted = [e for e in embeddings[0]]

        outputs = []
        for step in range(self.num_steps):
            # choose random index of embeddings (except for start/end tokens)
            if self.position == Position.END:
                random_idx = random.randint(
                    len(data_to_attack.transactions) - 2, max(1, len(adv_data.transactions) - 2)
                )
            else:
                raise NotImplementedError

            # only one embedding can be modified
            embeddings_splitted[random_idx].requires_grad = True

            # calculate the loss for current embeddings
            loss = self.classifier.forward_on_transaction_embeddings(
                transaction_embeddings=torch.stack(embeddings_splitted, dim=0).unsqueeze(0),
                mask=emb_out["mask"],
                amounts=adv_inputs["amounts"],
                label=adv_inputs["label"],
            )["loss"]
            loss.backward()

            # update the chosen embedding
            embeddings_splitted[random_idx] = (
                embeddings_splitted[random_idx] + self.epsilon * embeddings_splitted[random_idx].grad.data.sign()
            )
            self.classifier.zero_grad()

            # find the closest embedding for the modified one
            distances = torch.nn.functional.pairwise_distance(embeddings_splitted[random_idx], self.emb_layer)
            # we dont choose special tokens
            distances[self.special_indexes] = 10 ** 16

            # swap embeddings
            closest_idx = distances.argmin().item()
            embeddings_splitted[random_idx] = self.emb_layer[closest_idx]
            embeddings_splitted = [e.detach() for e in embeddings_splitted]

            # get adversarial indexes
            adversarial_idexes = orig_indexes.clone()
            adversarial_idexes[0, random_idx] = closest_idx

            adv_data = deepcopy(data_to_attack)
            adv_data.transactions = decode_indexes(adversarial_idexes[0], vocab=self.classifier.vocab)

            adversarial_inputs = data_to_tensors(adv_data, self.reader, self.classifier.vocab, self.device)

            # get adversarial probability and adversarial label
            adv_probs = self.get_clf_probs(adversarial_inputs)
            adv_data.label = self.probs_to_label(adv_probs)
            adv_prob = adv_probs[self.label_to_index(data_to_attack.label)].item()

            output = AttackerOutput(
                data=data_to_attack.to_dict(),
                adversarial_data=adv_data.to_dict(),
                probability=orig_prob,
                adversarial_probability=adv_prob,
                prob_diff=(orig_prob - adv_prob),
                wer=word_error_rate_on_sequences(data_to_attack.transactions, adv_data.transactions),
            )
            outputs.append(output)

        best_output = self.find_best_attack(outputs)

        return best_output
