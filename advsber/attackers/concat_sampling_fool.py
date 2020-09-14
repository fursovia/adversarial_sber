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


class Position(str, Enum):
    START = "start"
    END = "end"


@Attacker.register("concat_sampling_fool")
class ConcatSamplingFool(SamplingFool):
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

    @torch.no_grad()
    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:
        inputs_to_attack = data_to_tensors(data_to_attack, self.reader, self.lm_model.vocab, self.device)

        orig_prob = self.get_clf_probs(inputs_to_attack)[data_to_attack.label].item()

        adv_data = deepcopy(data_to_attack)
        amounts = generate_transaction_amounts(self.total_amount, self.num_tokens_to_add)
        if self.position == Position.END:
            adv_data.transactions = adv_data.transactions + [MASK_TOKEN] * self.num_tokens_to_add
            adv_data.amounts = adv_data.amounts + amounts
        elif self.position == Position.START:
            adv_data.transactions = [MASK_TOKEN] * self.num_tokens_to_add + adv_data.transactions
            adv_data.amounts = amounts + adv_data.amounts
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
                data_to_attack.transactions +
                decode_indexes(idx, self.lm_model.vocab, drop_start_end=False) for idx in indexes
            ]
        elif self.position == Position.START:
            adversarial_sequences = [
                decode_indexes(idx, self.lm_model.vocab, drop_start_end=False) +
                data_to_attack.transactions for idx in indexes
            ]
        else:
            raise NotImplementedError

        outputs = []
        for adv_sequence in adversarial_sequences:
            adv_data.transactions = adv_sequence
            adv_inputs = data_to_tensors(adv_data, self.reader, self.lm_model.vocab, self.device)

            adv_probs = self.get_clf_probs(adv_inputs)
            adv_label = self.probs_to_label(adv_probs)
            adv_data.label = adv_label

            adv_prob = adv_probs[data_to_attack.label].item()

            output = AttackerOutput(
                data=data_to_attack.to_dict(),
                adversarial_data=adv_data.to_dict(),
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
