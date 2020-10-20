from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from dataclasses import dataclass
from dataclasses_json import dataclass_json
from allennlp.common.registrable import Registrable
from allennlp.models import Model
import torch

from advsber.settings import TransactionsData, ModelsInput, START_TOKEN, END_TOKEN, MASK_TOKEN
from advsber.dataset_readers.transactions_reader import TransactionsDatasetReader


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

    SPECIAL_TOKENS = ("@@UNKNOWN@@", "@@PADDING@@", START_TOKEN, END_TOKEN, MASK_TOKEN)

    def __init__(self, classifier: Model, reader: TransactionsDatasetReader, device: int = -1,) -> None:
        self.classifier = classifier
        self.classifier.eval()
        self.reader = reader
        self.vocab = self.classifier.vocab

        self.device = device
        if self.device >= 0 and torch.cuda.is_available():
            self.classifier.cuda(self.device)

        self.special_indexes = [self.vocab.get_token_index(token, "transactions") for token in self.SPECIAL_TOKENS]

    @abstractmethod
    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:
        pass

    def attack_from_tensors(self, tensor_data_to_attack: ModelsInput) -> AttackerOutput:
        # ModelsInput -> TransactionsData
        # and then self.attack(data)
        pass

    def get_clf_probs(self, inputs: ModelsInput) -> torch.Tensor:
        probs = self.classifier(**inputs)["probs"][0]
        return probs

    def probs_to_label(self, probs: torch.Tensor) -> int:
        label_idx = probs.argmax().item()
        label = self.index_to_label(label_idx)
        return label

    def index_to_label(self, label_idx: int) -> int:
        label = self.vocab.get_index_to_token_vocabulary("labels").get(label_idx, str(label_idx))
        return int(label)

    def label_to_index(self, label: int) -> int:
        label_idx = self.vocab.get_token_to_index_vocabulary("labels").get(str(label), label)
        return label_idx

    @staticmethod
    def find_best_attack(outputs: List[AttackerOutput]) -> AttackerOutput:
        if len(outputs) == 1:
            return outputs[0]

        changed_label_outputs = []
        for output in outputs:
            if output.data["label"] != output.adversarial_data["label"] and output.wer > 0:
                changed_label_outputs.append(output)

        if changed_label_outputs:
            sorted_outputs = sorted(changed_label_outputs, key=lambda x: x.prob_diff, reverse=True)
            best_output = min(sorted_outputs, key=lambda x: x.wer)
        else:
            best_output = max(outputs, key=lambda x: x.prob_diff)

        return best_output
