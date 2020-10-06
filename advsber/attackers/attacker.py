from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod


from dataclasses import dataclass
from dataclasses_json import dataclass_json
from allennlp.common.registrable import Registrable
from allennlp.models import Model
import torch

from advsber.settings import TransactionsData
from advsber.dataset_readers import TransactionsDatasetReader


@dataclass_json
@dataclass
class AttackerOutput:
    data: TransactionsData
    adversarial_data_target: TransactionsData
    probability_target: float  # original probability
    adversarial_probability_target: float
    prob_diff_target: float
    wer: int
    adversarial_probability_subst: float
    adversarial_data_subst: TransactionsData
    prob_diff_subst: float
    probability_subst: float
    history: Optional[List[Dict[str, Any]]] = None


class Attacker(ABC, Registrable):
    def __init__(
        self,
        classifier_target: Model,
        reader: TransactionsDatasetReader,
        device: int = -1,
        classifier_subst: Optional[Model] = None,
    ) -> None:
        self.classifier_target = classifier_target
        self.classifier_subst = classifier_subst
        self.classifier_subst.eval()
        self.classifier_target.eval()
        self.reader = reader

        self.device = device
        if self.device >= 0 and torch.cuda.is_available():
            self.classifier_subst.cuda(self.device)
            self.classifier_target.cuda(self.device)

    @abstractmethod
    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:
        pass

    def get_clf_probs_target(self, inputs) -> torch.Tensor:
        probs_target = self.classifier_target(**inputs)["probs"][0]
        return probs_target

    def get_clf_probs_subst(self, inputs) -> torch.Tensor:
        probs_subst = self.classifier_subst(**inputs)["probs"][0]
        return probs_subst

    def probs_to_label_target(self, probs_target: torch.Tensor) -> int:
        label_idx_target = probs_target.argmax().item()
        label_target = self.index_to_label_target(label_idx_target)
        return label_target

    def probs_to_label_subst(self, probs_subst: torch.Tensor) -> int:
        label_idx_subst = probs_subst.argmax().item()
        label_subst = self.index_to_label_subst(label_idx_subst)
        return label_subst

    def index_to_label_subst(self, label_idx: int) -> int:
        label = self.classifier_subst.vocab.get_index_to_token_vocabulary("labels").get(label_idx, str(label_idx))
        return int(label)

    def index_to_label_target(self, label_idx: int) -> int:
        label = self.classifier_target.vocab.get_index_to_token_vocabulary("labels").get(label_idx, str(label_idx))
        return int(label)

    def label_to_index_target(self, label: int) -> int:
        label_idx = self.classifier_target.vocab.get_token_to_index_vocabulary("labels").get(str(label), label)
        return label_idx

    def label_to_index_subst(self, label: int) -> int:
        label_idx = self.classifier_subst.vocab.get_token_to_index_vocabulary("labels").get(str(label), label)
        return label_idx

    @staticmethod
    def find_best_attack(outputs: List[AttackerOutput]) -> AttackerOutput:
        if len(outputs) == 1:
            return outputs[0]
        changed_label_outputs = []
        for output in outputs:
            if output.data["label"] != output.adversarial_data_subst["label"] and output.wer > 0:
                changed_label_outputs.append(output)
        if changed_label_outputs:
            sorted_outputs = sorted(changed_label_outputs, key=lambda x: x.prob_diff_subst, reverse=True)
            best_output = min(sorted_outputs, key=lambda x: x.wer)
        else:
            best_output = max(outputs, key=lambda x: x.prob_diff_subst)
        return best_output
