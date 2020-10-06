from copy import deepcopy
import random

import torch
from allennlp.models import Model
from allennlp.nn import util
from typing import Optional
from advsber.attackers.attacker import Attacker, AttackerOutput
from advsber.settings import TransactionsData, START_TOKEN, END_TOKEN, MASK_TOKEN
from advsber.dataset_readers.transactions_reader import TransactionsDatasetReader
from advsber.utils.data import data_to_tensors, decode_indexes
from advsber.utils.metrics import word_error_rate_on_sequences


@Attacker.register("fgsm")
class FGSM(Attacker):

    SPECIAL_TOKENS = ("@@UNKNOWN@@", "@@PADDING@@", START_TOKEN, END_TOKEN, MASK_TOKEN)

    def __init__(
        self,
        classifier_target: Model,  # TransactionsClassifier
        reader: TransactionsDatasetReader,
        num_steps: int = 10,
        epsilon: float = 0.01,
        device: int = -1,
        classifier_subst: Optional[Model] = None,
    ) -> None:
        super().__init__(classifier_target=classifier_target, classifier_subst=classifier_subst,
                         reader=reader, device=device)
        self.num_steps = num_steps
        self.epsilon = epsilon
        self.classifier_subst.train()
        self.emb_layer = util.find_embedding_layer(self.classifier_subst).weight
        self.special_indexes = [
            self.classifier_target.vocab.get_token_index(token, "transactions") for token in self.SPECIAL_TOKENS
        ]

    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:
        # get inputs to the model
        inputs = data_to_tensors(data_to_attack, reader=self.reader, vocab=self.classifier_target.vocab,
                                 device=self.device)

        # get original indexes of a sequence
        orig_indexes = inputs["transactions"]["tokens"]["tokens"]

        # original probability of the true label
        orig_prob_target = self.get_clf_probs_target(inputs)[self.label_to_index_target(data_to_attack.label)].item()
        orig_prob_subst = self.get_clf_probs_subst(inputs)[self.label_to_index_subst(data_to_attack.label)].item()
        # get mask and transaction embeddings
        emb_out = self.classifier_subst.get_transaction_embeddings(transactions=inputs["transactions"])

        # disable gradients using a trick
        embeddings = emb_out["transaction_embeddings"].detach()
        embeddings_splitted = [e for e in embeddings[0]]

        outputs = []
        for step in range(self.num_steps):
            # choose random index of embeddings (except for start/end tokens)
            random_idx = random.randint(1, max(1, len(data_to_attack.transactions) - 2))
            # only one embedding can be modified
            embeddings_splitted[random_idx].requires_grad = True

            # calculate the loss for current embeddings
            loss = self.classifier_subst.forward_on_transaction_embeddings(
                transaction_embeddings=torch.stack(embeddings_splitted, dim=0).unsqueeze(0),
                mask=emb_out["mask"],
                amounts=inputs["amounts"],
                label=inputs["label"],
            )["loss"]
            loss.backward()

            # update the chosen embedding
            embeddings_splitted[random_idx] = (
                embeddings_splitted[random_idx] + self.epsilon * embeddings_splitted[random_idx].grad.data.sign()
            )
            self.classifier_subst.zero_grad()

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
            adv_data.transactions = decode_indexes(adversarial_idexes[0], vocab=self.classifier_target.vocab)

            adv_inputs = data_to_tensors(adv_data, self.reader, self.classifier_target.vocab, self.device)

            # get adversarial probability and adversarial label
            adv_probs_target = self.get_clf_probs_target(adv_inputs)
            adv_probs_subst = self.get_clf_probs_subst(adv_inputs)
            adv_data_subst = adv_data
            adv_data_target = adv_data
            adv_data_target.label = self.probs_to_label_target(adv_probs_target)
            adv_prob_target = adv_probs_target[self.label_to_index_target(data_to_attack.label)].item()
            adv_data_subst.label = self.probs_to_label_subst(adv_probs_subst)
            adv_prob_subst = adv_probs_subst[self.label_to_index_subst(data_to_attack.label)].item()
            output = AttackerOutput(
                data=data_to_attack.to_dict(),
                adversarial_data_target=adv_data_target.to_dict(),
                probability_target=orig_prob_target,
                probability_subst=orig_prob_subst,
                adversarial_probability_target=adv_prob_target,
                prob_diff_target=(orig_prob_target - adv_prob_target),
                wer=word_error_rate_on_sequences(data_to_attack.transactions, adv_data_target.transactions),
                adversarial_data_subst=adv_data_subst.to_dict(),
                adversarial_probability_subst=adv_prob_subst,
                prob_diff_subst=(orig_prob_subst - adv_prob_subst),
            )
            outputs.append(output)

        best_output = self.find_best_attack(outputs)
        return best_output
