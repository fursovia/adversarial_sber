from enum import Enum
from copy import deepcopy
from typing import List, Optional, Dict, Any
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
        classifier_target: Model,
        reader: TransactionsDatasetReader,
        position: Position = Position.END,
        num_tokens_to_add: int = 2,
        total_amount: float = 5000,
        num_samples: int = 100,
        temperature: float = 1.0,
        device: int = -1,
        classifier_subst: Optional[Model] = None,
    ) -> None:
        super().__init__(
            masked_lm=masked_lm,
            classifier_target=classifier_target,
            classifier_subst = classifier_subst,
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
        orig_prob_target = self.get_clf_probs_target(inputs_to_attack)[self.label_to_index_target(data_to_attack.label)].item()
        orig_prob_subst = self.get_clf_probs_subst(inputs_to_attack)[self.label_to_index_subst(data_to_attack.label)].item()
        adv_data_target = deepcopy(data_to_attack)
        amounts = generate_transaction_amounts(self.total_amount, self.num_tokens_to_add)
        if self.position == Position.END:
            adv_data_target.transactions = adv_data_target.transactions + [MASK_TOKEN] * self.num_tokens_to_add
            adv_data_target.amounts = adv_data_target.amounts + amounts
        elif self.position == Position.START:
            adv_data_target.transactions = [MASK_TOKEN] * self.num_tokens_to_add + adv_data_target.transactions
            adv_data_target.amounts = amounts + adv_data_target.amounts
        else:
            raise NotImplementedError
        adv_data_subst = adv_data_target
        adv_inputs = data_to_tensors(adv_data_target, self.reader, self.lm_model.vocab, self.device)

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
            adv_data_target.transactions = adv_sequence
            adv_inputs = data_to_tensors(adv_data_target, self.reader, self.lm_model.vocab, self.device)
            adv_probs_target = self.get_clf_probs_target(adv_inputs)
            adv_label_target = self.probs_to_label_target(adv_probs_target)
            adv_data_target.label = adv_label_target
            adv_prob_target = adv_probs_target[self.label_to_index_target(data_to_attack.label)].item()
            if self.classifier_subst is not None:
                adv_probs_subst = self.get_clf_probs_subst(adv_inputs)
                adv_label_subst = self.probs_to_label_subst(adv_probs_subst)
                adv_data_subst.label = adv_label_subst
                output = AttackerOutput(
                    data=data_to_attack.to_dict(),
                    adversarial_data_target=adv_data_target.to_dict(),
                    probability_target=orig_prob_target,
                    probability_subst=orig_prob_subst,
                    adversarial_probability_target=adv_prob_target,
                    prob_diff_target=(orig_prob_target - adv_prob_target),
                    wer=word_error_rate_on_sequences(data_to_attack.transactions, adv_data_target.transactions),
                    adversarial_data_subst = adv_data_subst.to_dict(),
                    adversarial_probability_subst = adv_probs_subst,
                    prob_diff_subst=(orig_prob_subst - adv_probs_subst),
                )
            else:
                output = AttackerOutput(
                    data=data_to_attack.to_dict(),
                    adversarial_data_target=adv_data_target.to_dict(),
                    probability_target=orig_prob_target,
                    adversarial_probability_target=adv_prob_target,
                    prob_diff_target=(orig_prob_target - adv_prob_target),
                    wer=word_error_rate_on_sequences(data_to_attack.transactions, adv_data_target.transactions)
                )
            outputs.append(output)

        best_output = self.find_best_attack(outputs)
        # we don't need history here actually
        # best_output.history = [deepcopy(o.__dict__) for o in outputs]
        return best_output
