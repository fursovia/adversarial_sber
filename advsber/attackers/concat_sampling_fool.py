from enum import Enum

import torch
from torch.distributions import Categorical

from advsber.attackers.sampling_fool import SamplingFool
from advsber.attackers.attacker import AttackerOutput, Attacker
from advsber.utils.data import sequence_to_tensors, decode_indexes, MASK_TOKEN
from advsber.utils.metrics import word_error_rate


class Position(str, Enum):
    START = "start"
    END = "end"


@Attacker.register("concat_sampling_fool")
class ConcatSamplingFool(SamplingFool):
    def __init__(
        self,
        masked_lm_archive_path: str,
        classifier_archive_path: str,
        position: Position = Position.END,
        num_tokens_to_add: int = 2,
        num_samples: int = 100,
        temperature: float = 1.0,
        device: int = -1,
    ) -> None:
        super().__init__(
            masked_lm_archive_path=masked_lm_archive_path,
            classifier_archive_path=classifier_archive_path,
            num_samples=num_samples,
            temperature=temperature,
            device=device,
        )
        self.position = position
        self.num_tokens_to_add = num_tokens_to_add

    @torch.no_grad()
    def attack(self, sequence_to_attack: str, label_to_attack: int,) -> AttackerOutput:
        original_sequence = sequence_to_attack

        if self.position == Position.END:
            sequence_to_attack = original_sequence + " " + " ".join([MASK_TOKEN] * self.num_tokens_to_add)
        elif self.position == Position.START:
            sequence_to_attack = " ".join([MASK_TOKEN] * self.num_tokens_to_add) + " " + original_sequence
        else:
            raise NotImplementedError

        orig_prob = self.calculate_probs(original_sequence)[label_to_attack].item()

        inputs = sequence_to_tensors(sequence_to_attack, self.reader, self.lm_model.vocab, self.device)
        logits = self.lm_model(inputs)["logits"]

        # drop start and end tokens
        logits = logits[:, 1:-1]

        if self.position == Position.END:
            logits_to_sample = logits[:, -self.num_tokens_to_add :][0]
        elif self.position == Position.START:
            logits_to_sample = logits[:, : self.num_tokens_to_add][0]
        else:
            raise NotImplementedError

        indexes = Categorical(logits=logits_to_sample / self.temperature).sample((self.num_samples,))

        if self.position == Position.END:
            adversarial_sequences = [
                original_sequence + " " + decode_indexes(idx, self.lm_model.vocab) for idx in indexes
            ]
        elif self.position == Position.START:
            adversarial_sequences = [
                decode_indexes(idx, self.lm_model.vocab) + " " + original_sequence for idx in indexes
            ]
        else:
            raise NotImplementedError

        outputs = []
        for adv_sequence in adversarial_sequences:
            adv_probs = self.calculate_probs(adv_sequence)
            adv_prob = adv_probs[label_to_attack].item()
            output = AttackerOutput(
                sequence=original_sequence,
                adversarial_sequence=adv_sequence,
                probability=orig_prob,
                adversarial_probability=adv_prob,
                attacked_label=label_to_attack,
                adversarial_label=adv_probs.argmax().item(),
                wer=word_error_rate(original_sequence, adv_sequence),
                prob_diff=(orig_prob - adv_prob),
            )
            outputs.append(output)

        best_output = self.find_best_attack(outputs)
        # we don't need history here actually
        # best_output.history = [deepcopy(o.__dict__) for o in outputs]
        return best_output
