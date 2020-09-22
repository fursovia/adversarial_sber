from enum import Enum
from copy import deepcopy

import torch
from torch.distributions import Categorical
from allennlp.models import Model

from advsber.attackers.sampling_fool import SamplingFool
from advsber.utils.data import decode_indexes, data_to_tensors, generate_transaction_amounts
from advsber.settings import MASK_TOKEN
from advsber.utils.metrics import word_error_rate_on_sequences
from advsber.dataset_readers.transactions_reader import TransactionsDatasetReader
from advsber.attackers.attacker import Attacker, AttackerOutput
from advsber.settings import TransactionsData
from advsber.attackers.concat_sampling_fool import ConcatSamplingFool
from advsber.attackers.concat_sampling_fool import Position


@Attacker.register("greedy_concat_sampling_fool")
class GreedyConcatSamplingFool(Attacker):
    def __init__(
            self,
            masked_lm: Model,
            classifier: Model,
            reader: TransactionsDatasetReader,
            position: Position = Position.END,
            num_tokens_to_add: int = 2,
            total_amount: float = 5000,
            num_samples: int = 100,
            temperature: float = 1.0,
            device: int = -1,
    ) -> None:
        super().__init__(classifier = classifier,
                         reader = reader,
                         device = device)

        self.total_amount = total_amount
        self.num_tokens_to_add = num_tokens_to_add

        self.attacker = ConcatSamplingFool(masked_lm=masked_lm,
                                        classifier=classifier,
                                        reader=reader,
                                        position=position,
                                        num_tokens_to_add=1,
                                        total_amount=0.0,
                                        num_samples=num_samples,
                                        temperature=temperature,
                                        device=device
                                    )

    @torch.no_grad()
    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:
        inputs_to_attack = data_to_tensors(data_to_attack, self.reader, self.attacker.lm_model.vocab, self.device)
        orig_prob = self.get_clf_probs(inputs_to_attack)[self.label_to_index(data_to_attack.label)].item()

        adv_data = deepcopy(data_to_attack)
        amounts = generate_transaction_amounts(self.total_amount, self.num_tokens_to_add)

        for amount in amounts:
            self.attacker.total_amount = amount
            adv_data = data_to_tensors(adv_data, self.reader, self.attacker.lm_model.vocab, self.device)
            output = self.attacker.attack(adv_data)
            adv_data = output.to_dict()['data']
            adv_data = TransactionsData(**adv_data)

        adv_inputs = data_to_tensors(adv_data, self.reader, self.lm_model.vocab, self.device)
        adv_probs = self.get_clf_probs(adv_inputs)
        adv_prob = adv_probs[self.label_to_index(data_to_attack.label)].item()

        output = AttackerOutput(
            data=data_to_attack.to_dict(),
            adversarial_data=adv_data.to_dict(),
            probability=orig_prob,
            adversarial_probability=adv_prob,
            prob_diff=(orig_prob - adv_prob),
            wer=word_error_rate_on_sequences(data_to_attack.transactions, adv_data.transactions),
        )
        return output

