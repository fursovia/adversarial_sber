from copy import deepcopy
from typing import List, Optional, Dict, Any
import torch
from torch.distributions import Categorical
from allennlp.models import Model

from advsber.attackers.attacker import Attacker, AttackerOutput
from advsber.settings import TransactionsData
from advsber.utils.data import decode_indexes, data_to_tensors
from advsber.utils.metrics import word_error_rate_on_sequences
from advsber.dataset_readers.transactions_reader import TransactionsDatasetReader


@Attacker.register("sampling_fool")
class SamplingFool(Attacker):
    """
    SamplingFool samples sequences using Masked LM
    """

    def __init__(
        self,
        masked_lm: Model,
        classifier_target: Model,
        reader: TransactionsDatasetReader,
        num_samples: int = 100,
        temperature: float = 1.0,
        device: int = -1,
        classifier_subst: Optional[Model] = None,
    ) -> None:
        super().__init__(classifier_target=classifier_target, classifier_subst = classifier_subst, reader=reader, device=device)
        self.lm_model = masked_lm
        # disable masker by hands
        self.lm_model._tokens_masker = None
        self.lm_model.eval()

        if self.device >= 0 and torch.cuda.is_available():
            self.lm_model.cuda(self.device)

        self.num_samples = num_samples
        self.temperature = temperature

    def get_lm_logits(self, inputs) -> torch.Tensor:
        logits = self.lm_model(**inputs)["logits"]
        return logits

    @torch.no_grad()
    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:
        inputs_to_attack = data_to_tensors(data_to_attack, self.reader, self.lm_model.vocab, self.device)

        orig_prob_target = self.get_clf_probs_target(inputs_to_attack)[self.label_to_index_target(data_to_attack.label)].item()
        orig_prob_subst = self.get_clf_probs_subst(inputs_to_attack)[self.label_to_index_subst(data_to_attack.label)].item()
        logits = self.get_lm_logits(inputs_to_attack)
        indexes = Categorical(logits=logits[0] / self.temperature).sample((self.num_samples,))
        adversarial_sequences = [decode_indexes(idx, self.lm_model.vocab) for idx in indexes]
        outputs = []
        adv_data_target = deepcopy(data_to_attack)
        adv_data_subst = deepcopy(data_to_attack)
        for adv_sequence in adversarial_sequences:
            adv_data_target.transactions = adv_sequence
            adv_data_subst.transactions = adv_sequence
            adv_inputs = data_to_tensors(adv_data_target, self.reader, self.lm_model.vocab, self.device)
            adv_probs_target = self.get_clf_probs_target(adv_inputs)
            adv_label_target = self.probs_to_label_target(adv_probs_target)
            adv_data_target.label = adv_label_target
            adv_prob_target = adv_probs_target[self.label_to_index_target(data_to_attack.label)].item()
            if self.classifier_subst is not None:
                adv_probs_subst = self.get_clf_probs_subst(adv_inputs)
                adv_label_subst = self.probs_to_label_subst(adv_probs_target)
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
