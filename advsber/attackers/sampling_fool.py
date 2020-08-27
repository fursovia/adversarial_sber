from pathlib import Path
from copy import deepcopy

import torch
from torch.distributions import Categorical
from allennlp.models import Model, load_archive
from allennlp.data import DatasetReader

from advsber.attackers.attacker import Attacker, AttackerOutput
from advsber.utils.data import sequence_to_tensors, decode_indexes
from advsber.utils.distance import calculate_wer


@Attacker.register("sampling_fool")
class SamplingFool(Attacker):
    """
    SamplingFool samples sequences using Masked LM
    """

    def __init__(
            self,
            masked_lm_dir: str,
            classifier_dir: str,
            num_samples: int = 100,
            temperature: float = 1.0,
            device: int = -1
    ) -> None:

        archive = load_archive(Path(masked_lm_dir) / "model.tar.gz")
        self.lm_model = archive.model
        # disable masker by hands
        self.lm_model._tokens_masker = None
        self.lm_model.eval()

        self.classifier = Model.from_archive(Path(classifier_dir) / "model.tar.gz")
        self.classifier.eval()

        self.reader = DatasetReader.from_params(archive.config["dataset_reader"])

        self.num_samples = num_samples
        self.temperature = temperature

        self.device = device
        if self.device >= 0 and torch.cuda.is_available():
            self.lm_model.cuda(self.device)
            self.classifier.cuda(self.device)

    def calculate_probs(self, sequence: str) -> torch.Tensor:
        inputs = sequence_to_tensors(sequence, self.reader, self.lm_model.vocab, self.device)
        probs = self.classifier(inputs)["probs"][0]
        return probs

    @torch.no_grad()
    def attack(
            self,
            sequence_to_attack: str,
            label_to_attack: int,
    ) -> AttackerOutput:
        orig_prob = self.calculate_probs(sequence_to_attack)[label_to_attack].item()

        inputs = sequence_to_tensors(sequence_to_attack, self.reader, self.lm_model.vocab, self.device)
        logits = self.lm_model(inputs)["logits"]
        indexes = Categorical(
            logits=logits[0] / self.temperature
        ).sample((self.num_samples,))

        adversarial_sequences = [decode_indexes(idx, self.lm_model.vocab) for idx in indexes]

        outputs = []
        for adv_sequence in adversarial_sequences:
            adv_probs = self.calculate_probs(adv_sequence)
            adv_prob = adv_probs[label_to_attack].item()
            output = AttackerOutput(
                sequence=sequence_to_attack,
                adversarial_sequence=adv_sequence,
                probability=orig_prob,
                adversarial_probability=adv_prob,
                attacked_label=label_to_attack,
                adversarial_label=adv_probs.argmax().item(),
                wer=calculate_wer(sequence_to_attack, adv_sequence),
                prob_diff=(orig_prob - adv_prob)
            )
            outputs.append(output)

        best_output = self.find_best_attack(outputs)
        best_output.history = [deepcopy(o.__dict__) for o in outputs]
        return best_output
