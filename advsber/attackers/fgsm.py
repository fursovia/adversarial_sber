from copy import deepcopy
import random

import torch
from allennlp.models import Model

from advsber.attackers.attacker import Attacker, AttackerOutput
from advsber.settings import TransactionsData
from advsber.dataset_readers.transactions_reader import TransactionsDatasetReader
from advsber.utils.data import data_to_tensors


@Attacker.register("fgsm")
class FGSM(Attacker):
    def __init__(
        self,
        classifier: Model,  # TransactionsClassifier
        reader: TransactionsDatasetReader,
        num_steps: int = 10,
        epsilon: float = 0.01,
        device: int = -1,
    ) -> None:
        super().__init__(classifier=classifier, reader=reader, device=device)
        self.num_steps = num_steps
        self.epsilon = epsilon

    @torch.no_grad()
    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:
        seq_length = len(data_to_attack.transactions)

        inputs = data_to_tensors(data_to_attack, reader=self.reader, vocab=self.classifier.vocab, device=self.device)

        emb_out = self.classifier.get_transaction_embeddings(inputs["transaction"])
        embs = emb_out["transaction_embeddings"].detach()

        orig_probs = self.get_clf_probs(inputs)
        label = self.label_to_index(data_to_attack.label)
        orig_prob = orig_probs[label].item()

        embs = [e for e in embs[0]]
        for step in range(self.num_steps):
            random_idx = random.randint(1, max(1, seq_length - 2))
            embs[random_idx].requires_grad = True
            embeddings_tensor = torch.stack(embs, dim=0).unsqueeze(0)

            clf_output = self.classifier.forward_on_transaction_embeddings(
                transaction_embeddings=embeddings_tensor,
                mask=emb_out["mask"],
                amounts=inputs["amounts"],
                label=inputs["label"],
            )

            loss = clf_output["loss"]
            loss.backward()
            self.classifier.zero_grad()

            embs[random_idx] = embs[random_idx] + self.epsilon * embs[random_idx].grad.data.sign()
            distances = torch.nn.functional.pairwise_distance(embs[random_idx], self.emb_layer)

            # @UNK@, @PAD@, @MASK@, @START@, @END@
            to_drop_indexes = [0, 1] + list(range(self.vocab_size - 3, self.vocab_size))
            distances[to_drop_indexes] = 10e6

        adv_data = deepcopy(data_to_attack)
        output = AttackerOutput(
            data=data_to_attack.to_dict(),
            adversarial_data=adv_data.to_dict(),
            probability=0.5,
            adversarial_probability=0.4,
            prob_diff=0.1,
            wer=5,
        )

        return output
