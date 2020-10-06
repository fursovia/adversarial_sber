from copy import deepcopy

import torch
from allennlp.models import Model
from typing import Optional
from advsber.utils.data import data_to_tensors, generate_transaction_amounts
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
        super().__init__(classifier_target=classifier_target, classifier_subst=classifier_subst,
                         reader=reader,  device=device)

        self.total_amount = total_amount
        self.num_tokens_to_add = num_tokens_to_add

        self.attacker = ConcatSamplingFool(
            masked_lm=masked_lm,
            classifier_target=classifier_target,
            classifier_subst=classifier_subst,
            reader=reader,
            position=position,
            num_tokens_to_add=1,
            total_amount=0.0,
            num_samples=num_samples,
            temperature=temperature,
            device=device,
        )

    @torch.no_grad()
    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:
        inputs_to_attack = data_to_tensors(data_to_attack, self.reader, self.attacker.lm_model.vocab, self.device)
        orig_prob_subst = \
            self.get_clf_probs_subst(inputs_to_attack)[self.label_to_index_subst(data_to_attack.label)].item()
        orig_prob_target = \
            self.get_clf_probs_target(inputs_to_attack)[self.label_to_index_target(data_to_attack.label)].item()
        adv_data_target = deepcopy(data_to_attack)
        adv_data_subst = deepcopy(data_to_attack)
        amounts = generate_transaction_amounts(self.total_amount, self.num_tokens_to_add)
        for amount in amounts:
            self.attacker.total_amount = amount
            output = self.attacker.attack(adv_data_subst)
            adv_data_target = output.to_dict()["adversarial_data_target"]
            adv_data_target = TransactionsData(**adv_data_target)
            adv_data_subst = output.to_dict()["adversarial_data_subst"]
            adv_data_subst = TransactionsData(**adv_data_subst)

        adv_inputs = data_to_tensors(adv_data_subst, self.reader, self.attacker.lm_model.vocab, self.device)
        adv_probs_target = self.get_clf_probs_target(adv_inputs)
        adv_probs_subst = self.get_clf_probs_subst(adv_inputs)
        adv_prob_target = adv_probs_target[self.label_to_index_target(data_to_attack.label)].item()
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
        return output