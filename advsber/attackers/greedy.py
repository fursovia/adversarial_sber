from copy import deepcopy

import torch
import numpy as np
from allennlp.models import Model

from advsber.utils.data import data_to_tensors
from advsber.utils.metrics import word_error_rate_on_sequences
from advsber.dataset_readers.transactions_reader import TransactionsDatasetReader
from advsber.attackers.attacker import Attacker, AttackerOutput
from advsber.settings import TransactionsData


@Attacker.register("greedy")
class GreedyAttacker(Attacker):
    def __init__(
        self, classifier: Model, reader: TransactionsDatasetReader, num_steps: int, device: int = -1,
    ) -> None:
        super().__init__(classifier=classifier, reader=reader, device=device)
        self._num_steps = num_steps

    def get_probability_of_data(self, data: TransactionsData) -> float:
        inputs_to_attack = data_to_tensors(data, self.reader, self.vocab, self.device)
        prob = self.get_clf_probs(inputs_to_attack)[self.label_to_index(data.label)].item()
        return prob

    @torch.no_grad()
    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:
        orig_prob = self.get_probability_of_data(data_to_attack)
        adv_data = deepcopy(data_to_attack)

        indexes_to_flip = np.random.randint(0, len(data_to_attack), size=self._num_steps)

        outputs = []
        for index_to_flip in indexes_to_flip:
            probabilities = {}

            for idx, token in self.vocab.get_index_to_token_vocabulary(namespace="transactions").items():
                curr_adv_data = deepcopy(adv_data)
                curr_adv_data.transactions[index_to_flip] = token
                curr_prob = self.get_probability_of_data(curr_adv_data)
                probabilities[token] = curr_prob

            probabilities_sorted = sorted(probabilities.items(), key=lambda x: x[1], reverse=False)
            max_token, adv_prob = probabilities_sorted[0]

            prob_drop = orig_prob - adv_prob
            if prob_drop > 0.0:
                adv_inputs = data_to_tensors(adv_data, self.reader, self.vocab, self.device)
                adv_data.transactions[index_to_flip] = max_token
                adv_data.label = self.probs_to_label(self.get_clf_probs(adv_inputs))

                output = AttackerOutput(
                    data=data_to_attack.to_dict(),
                    adversarial_data=adv_data.to_dict(),
                    probability=orig_prob,
                    adversarial_probability=adv_prob,
                    prob_diff=prob_drop,
                    wer=word_error_rate_on_sequences(data_to_attack.transactions, adv_data.transactions),
                )
                outputs.append(output)

        # TODO: empty outputs
        best_output = self.find_best_attack(outputs)
        return best_output
