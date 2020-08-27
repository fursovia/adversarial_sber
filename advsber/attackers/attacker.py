from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from dataclasses import dataclass
from allennlp.common.registrable import Registrable


@dataclass
class AttackerOutput:
    sequence: str  # attacked sequence
    adversarial_sequence: str
    probability: float  # original probability
    adversarial_probability: float
    attacked_label: int  # targeted attack
    adversarial_label: int  # predicted label
    wer: int
    prob_diff: float
    history: Optional[List[Dict[str, Any]]] = None


class Attacker(ABC, Registrable):
    @abstractmethod
    def attack(self, sequence_to_attack: str, label_to_attack: int) -> AttackerOutput:
        pass

    @staticmethod
    def find_best_attack(outputs: List[AttackerOutput]) -> AttackerOutput:
        if len(outputs) == 1:
            return outputs[0]

        changed_label_outputs = []
        for output in outputs:
            if output.attacked_label != output.adversarial_label and output.wer > 0:
                changed_label_outputs.append(output)

        if changed_label_outputs:
            sorted_outputs = sorted(changed_label_outputs, key=lambda x: x.prob_diff, reverse=True)
            best_output = min(sorted_outputs, key=lambda x: x.wer)
        else:
            best_output = max(outputs, key=lambda x: x.prob_diff)

        return best_output
