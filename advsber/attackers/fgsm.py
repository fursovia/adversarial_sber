from copy import deepcopy
import random
import typer
import torch
from allennlp.models import Model
from allennlp.nn import util
import tokenize
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers import WhitespaceTokenizer, Token
from advsber.attackers.attacker import Attacker, AttackerOutput
from advsber.settings import TransactionsData
from advsber.dataset_readers.transactions_reader import TransactionsDatasetReader
from advsber.utils.data import data_to_tensors, decode_indexes, decode_indexes_amounts
from advsber.utils.metrics import word_error_rate_on_sequences
from allennlp.data.fields import TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
import numpy as np
from operator import itemgetter


@Attacker.register("fgsm")
class FGSM(Attacker):
    def __init__(
        self,
        classifier: Model,  # TransactionsClassifier
        reader: TransactionsDatasetReader,
        num_steps: int = 10,
        epsilon: str = "1000",
        total_amount: str = "5",
        device: int = -1,
    ) -> None:
        super().__init__(classifier=classifier, reader=reader, device=device)
        self.classifier = self.classifier.train()
        self.num_steps = num_steps
        self.epsilon = float(epsilon)
        self.total_amount = float(total_amount)
        self.emb_layer = util.find_embedding_layer(self.classifier).weight

    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:
        # get inputs to the model
        voc_data = deepcopy(data_to_attack)
        inputs = data_to_tensors(data_to_attack, reader=self.reader, vocab=self.vocab, device=self.device)
        adversarial_idexes = inputs["amounts"]["tokens"]["tokens"][0]
        # original probability of the true label
        orig_prob = self.get_clf_probs(inputs)[self.label_to_index(data_to_attack.label)].item()
        # get mask and transaction embeddings
        emb_out = self.classifier.get_transaction_embeddings(transactions=inputs["transactions"])
        emb_amounts_out = self.classifier.get_amounts_embeddings(amounts=inputs["amounts"])
        cuda0 = torch.device("cuda:0")
        size = self.vocab.get_vocab_size(namespace="amounts")
        indexes = [i for i in range(0, size - 1)]
        voc = [self.vocab.get_token_from_index(idx, namespace="amounts") for idx in indexes]
        if "@@PADDING@@" in voc:
            st = voc.index("@@PADDING@@")
            voc[st] = "0"
        if "@@UNKNOWN@@" in voc:
            st = voc.index("@@UNKNOWN@@")
            voc[st] = "0"
        if "<START>" in voc:
            st = voc.index("<START>")
            voc[st] = "1"
        if "<END>" in voc:
            st = voc.index("<END>")
            voc[st] = "-1"
        voc_data.amounts = voc
        voc_inputs = data_to_tensors(
            voc_data, reader=self.reader, vocab=self.vocab, device=self.device, transform=False
        )
        emb_vocab_amounts = self.classifier.get_amounts_embeddings(amounts=voc_inputs["amounts"])
        # disable gradients using a trick
        embeddings_vocab_amounts = emb_vocab_amounts["amounts_embeddings"].detach()
        embeddings = emb_out["transaction_embeddings"].detach()
        embeddings_amounts = emb_amounts_out["amounts_embeddings"].detach()
        embeddings_splitted = [e for e in embeddings[0]]
        amounts_splitted = [torch.tensor(e, device=cuda0) for e in data_to_attack.amounts]
        embeddings_amounts_splitted = [e for e in embeddings_amounts[0]]
        embeddings_vocab_amounts_splitted = [e for e in embeddings_vocab_amounts[0]]
        outputs = []
        for step in range(self.num_steps):
            # choose random index of embeddings (except for start/end tokens)
            random_idx = random.randint(1, max(1, len(data_to_attack.amounts) - 2))
            # only one embedding can be modified
            embeddings_amounts_splitted[random_idx].requires_grad = True
            # calculate the loss for current embeddings
            loss = self.classifier.forward_on_transaction_embeddings(
                transaction_embeddings=torch.stack(embeddings_splitted, dim=0).unsqueeze(0),
                mask=emb_out["mask"],
                amount_embeddings=torch.stack(embeddings_amounts_splitted, dim=0).unsqueeze(0),
                label=inputs["label"],
            )["loss"]
            loss.backward()
            # update the chosen embedding
            embeddings_amounts_splitted[random_idx] = (
                embeddings_amounts_splitted[random_idx]
                + self.epsilon * embeddings_amounts_splitted[random_idx].grad.data.sign()
            )
            self.classifier.zero_grad()
            # find the closest embedding for the modified one

            embeddings_vocab_amounts = embeddings_vocab_amounts.squeeze(0)
            if data_to_attack.amounts[random_idx] >= 0:
                max_amount = self.reader.discretizer.transform(
                    [[data_to_attack.amounts[random_idx] + self.total_amount]]
                )[0][0]
                min_amount = self.reader.discretizer.transform([[data_to_attack.amounts[random_idx]]])[0][0]
                sign = 1
            else:
                max_amount = self.reader.discretizer.transform([[data_to_attack.amounts[random_idx]]])[0][0]
                min_amount = self.reader.discretizer.transform(
                    [[data_to_attack.amounts[random_idx] - self.total_amount]]
                )[0][0]
                sign = -1
            # for negative amounts in gender
            res_idx = [idx for idx, val in enumerate(voc) if int(val) <= max_amount and int(val) >= min_amount]
            res_voc = list(itemgetter(*res_idx)(voc))
            distances = torch.nn.functional.pairwise_distance(
                embeddings_amounts_splitted[random_idx], embeddings_vocab_amounts
            )
            # we dont choose special tokens
            # swap embeddings
            if min_amount == max_amount:
                closest_idx = 0
            else:
                res_distances = list(itemgetter(*res_idx)(distances))
                # closest_idx = distances.argmin().item()
                closest_idx = res_distances.index(min(res_distances))
            embeddings_amounts_splitted[random_idx] = embeddings_vocab_amounts[res_idx[closest_idx]]
            embeddings_amounts_splitted = [e.detach() for e in embeddings_amounts_splitted]
            # get adversarial indexes
            adv_data = deepcopy(data_to_attack)
            if voc[closest_idx] == "@@PADDING@@":
                voc[closest_idx] = 0
            if voc[closest_idx] == "@@UNKNOWN@@":
                voc[closest_idx] = 0
            if voc[closest_idx] == "<START>":
                voc[closest_idx] = 1
            if voc[closest_idx] == "<END>":
                voc[closest_idx] = -1
            if sign == 1:
                adv_data.amounts[random_idx] = max(
                    min(
                        self.reader.discretizer.inverse_transform([[res_voc[closest_idx]]])[0][0],
                        self.total_amount + data_to_attack.amounts[random_idx],
                    ),
                    data_to_attack.amounts[random_idx],
                )
            else:
                adv_data.amounts[random_idx] = min(
                    max(
                        self.reader.discretizer.inverse_transform([[res_voc[closest_idx]]])[0][0],
                        data_to_attack.amounts[random_idx] - self.total_amount,
                    ),
                    data_to_attack.amounts[random_idx],
                )

            adv_inputs = data_to_tensors(adv_data, self.reader, self.vocab, self.device)
            # get adversarial probability and adversarial label
            adv_probs = self.get_clf_probs(adv_inputs)
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
        best_output.history = [output.to_dict() for output in outputs]
        return best_output
