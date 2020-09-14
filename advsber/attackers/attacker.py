from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from dataclasses import dataclass
from dataclasses_json import dataclass_json
from allennlp.common.registrable import Registrable
import torch

from advsber.settings import TransactionsData
from advsber.models import TransactionsClassifier
from advsber.dataset_readers import TransactionsDatasetReader


@dataclass_json
@dataclass
class AttackerOutput:
    data: TransactionsData
    adversarial_data: TransactionsData
    probability: float  # original probability
    adversarial_probability: float
    prob_diff: float
    wer: int
    history: Optional[List[Dict[str, Any]]] = None


class Attacker(ABC, Registrable):

    def __init__(
        self,
        classifier: TransactionsClassifier,
        reader: TransactionsDatasetReader,
        device: int = -1,
    ) -> None:
        self.classifier = classifier
        self.classifier.eval()
        self.reader = reader

        self.device = device
        if self.device >= 0 and torch.cuda.is_available():
            self.classifier.cuda(self.device)

    @abstractmethod
    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:
        pass

    @staticmethod
    def find_best_attack(outputs: List[AttackerOutput]) -> AttackerOutput:
        if len(outputs) == 1:
            return outputs[0]

        changed_label_outputs = []
        for output in outputs:
            if output.data.label != output.adversarial_data.label and output.wer > 0:
                changed_label_outputs.append(output)

        if changed_label_outputs:
            sorted_outputs = sorted(changed_label_outputs, key=lambda x: x.prob_diff, reverse=True)
            best_output = min(sorted_outputs, key=lambda x: x.wer)
        else:
            best_output = max(outputs, key=lambda x: x.prob_diff)

        return best_output
