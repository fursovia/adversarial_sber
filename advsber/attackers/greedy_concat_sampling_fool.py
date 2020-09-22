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


class Position(str, Enum):
    START = "start"
    END = "end"


@Attacker.register("greedy_concat_sampling_fool")
class GreedyConcatSamplingFool(SamplingFool):
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
        super().__init__(
            masked_lm=masked_lm,
            classifier=classifier,
            reader=reader,
            num_samples=num_samples,
            temperature=temperature,
            device=device
        )
        self.position = position
        self.num_tokens_to_add = num_tokens_to_add
        self.total_amount = total_amount
        self.masked_lm = masked_lm
        self.classifier = classifier

    @torch.no_grad()
    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:

        adv_data = deepcopy(data_to_attack)
        amounts = generate_transaction_amounts(self.total_amount, self.num_tokens_to_add)

        for amount in amounts:
            attacker = ConcatSamplingFool(
                masked_lm=self.masked_lm,
                classifier=self.classifier,
                num_tokens_to_add=1,
                total_amount=amount
            )
            output = attacker.attack(adv_data)
            adv_data = output.to_dict()['data']
            adv_data = TransactionsData(**adv_data)

        return output

