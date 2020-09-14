from enum import Enum
from copy import deepcopy

import torch
from torch.distributions import Categorical

from advsber.attackers.sampling_fool import SamplingFool
from advsber.utils.data import decode_indexes, data_to_tensors
from advsber.settings import MASK_TOKEN
from advsber.utils.metrics import word_error_rate_on_sequences
from advsber.models.masked_lm import MaskedLanguageModel
from advsber.dataset_readers.transactions_reader import TransactionsDatasetReader
from advsber.models.classifier import TransactionsClassifier
from advsber.attackers.attacker import Attacker, AttackerOutput
from advsber.settings import TransactionsData


class Position(str, Enum):
    START = "start"
    END = "end"


@Attacker.register("concat_sampling_fool")
class ConcatSamplingFool(SamplingFool):
    def __init__(
        self,
        masked_lm: MaskedLanguageModel,
        classifier: TransactionsClassifier,
        reader: TransactionsDatasetReader,
        position: Position = Position.END,
        num_tokens_to_add: int = 2,
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

    @torch.no_grad()
    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:
        inputs_to_attack = data_to_tensors(data_to_attack, self.reader, self.lm_model.vocab, self.device)

        orig_prob = self.get_clf_probs(inputs_to_attack)[data_to_attack.label].item()

        adv_data = deepcopy(data_to_attack)
        if self.position == Position.END:
            adv_data.transactions = adv_data.transactions + [MASK_TOKEN] * self.num_tokens_to_add
        elif self.position == Position.START:
            adv_data.transactions = [MASK_TOKEN] * self.num_tokens_to_add + adv_data.transactions
        else:
            raise NotImplementedError

        adv_inputs = data_to_tensors(adv_data, self.reader, self.lm_model.vocab, self.device)

        logits = self.get_lm_logits(adv_inputs)
        # drop start and end tokens
        logits = logits[:, 1:-1]

        if self.position == Position.END:
            logits_to_sample = logits[:, -self.num_tokens_to_add:][0]
        elif self.position == Position.START:
            logits_to_sample = logits[:, :self.num_tokens_to_add][0]
        else:
            raise NotImplementedError

        indexes = Categorical(logits=logits_to_sample / self.temperature).sample((self.num_samples,))

        if self.position == Position.END:
            adversarial_sequences = [
                data_to_attack.transactions + decode_indexes(idx, self.lm_model.vocab) for idx in indexes
            ]
        elif self.position == Position.START:
            adversarial_sequences = [
                decode_indexes(idx, self.lm_model.vocab) + data_to_attack.transactions for idx in indexes
            ]
        else:
            raise NotImplementedError

        outputs = []
        for adv_sequence in adversarial_sequences:
            adv_data.transactions = adv_sequence
            adv_inputs = data_to_tensors(adv_data, self.reader, self.lm_model.vocab, self.device)

            adv_prob = self.calculate_probs(adv_inputs)[data_to_attack.label].item()

            output = AttackerOutput(
                data=data_to_attack,
                adversarial_data=adv_data,
                probability=orig_prob,
                adversarial_probability=adv_prob,
                prob_diff=(orig_prob - adv_prob),
                wer=word_error_rate_on_sequences(data_to_attack.transactions, adv_data.transactions)
            )
            outputs.append(output)

        best_output = self.find_best_attack(outputs)
        # we don't need history here actually
        # best_output.history = [deepcopy(o.__dict__) for o in outputs]
        return best_output
