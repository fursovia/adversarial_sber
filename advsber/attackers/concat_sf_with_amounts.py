from enum import Enum
from typing import Dict, Any

import torch
from torch.distributions import Categorical

from advsber.attackers.sampling_fool import SamplingFool
from advsber.attackers.attacker import AttackerOutput, Attacker
from advsber.utils.data import (
    data_to_tensors,
    decode_indexes,
    MASK_TOKEN,
    generate_transaction_amounts,
    load_discretizer,
)
from advsber.utils.metrics import word_error_rate


class Position(str, Enum):
    START = "start"
    END = "end"


@Attacker.register("concat_sampling_fool_with_amounts")
class ConcatSamplingFoolWithAmounts(SamplingFool):
    def __init__(
        self,
        masked_lm_archive_path: str,
        classifier_archive_path: str,
        discretizer_path: str,
        position: Position = Position.END,
        num_tokens_to_add: int = 2,
        total_amount: float = 5000,
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
        self.discretizer = load_discretizer(discretizer_path)
        self.total_amount = total_amount
        self.position = position
        self.num_tokens_to_add = num_tokens_to_add

    @torch.no_grad()
    def attack(self, data_to_attack: Dict[str, Any]) -> AttackerOutput:
        orig_transactions = data_to_attack["transactions"]
        orig_amounts = data_to_attack["amounts"]

        adversarial_amounts = generate_transaction_amounts(
            total_amount=self.total_amount,
            num_transactions=self.num_tokens_to_add
        )

        if self.position == Position.END:
            data_to_attack["transactions"] = orig_transactions + " " + " ".join([MASK_TOKEN] * self.num_tokens_to_add)
            data_to_attack["amounts"] = orig_amounts + adversarial_amounts
        elif self.position == Position.START:
            sequence_to_attack = " ".join([MASK_TOKEN] * self.num_tokens_to_add) + " " + orig_transactions
            data_to_attack["amounts"] = adversarial_amounts + orig_amounts
        else:
            raise NotImplementedError

        orig_prob = self.calculate_probs(orig_transactions)[data_to_attack["label"]].item()

        inputs = data_to_tensors(data_to_attack, self.reader, self.lm_model.vocab, self.device)
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
                orig_transactions + " " + decode_indexes(idx, self.lm_model.vocab) for idx in indexes
            ]
        elif self.position == Position.START:
            adversarial_sequences = [
                decode_indexes(idx, self.lm_model.vocab) + " " + orig_transactions for idx in indexes
            ]
        else:
            raise NotImplementedError

        outputs = []
        for adv_sequence in adversarial_sequences:
            adv_probs = self.calculate_probs(adv_sequence)
            adv_prob = adv_probs[label_to_attack].item()
            output = AttackerOutput(
                sequence=orig_transactions,
                adversarial_sequence=adv_sequence,
                probability=orig_prob,
                adversarial_probability=adv_prob,
                attacked_label=label_to_attack,
                adversarial_label=adv_probs.argmax().item(),
                wer=word_error_rate(orig_transactions, adv_sequence),
                prob_diff=(orig_prob - adv_prob),
            )
            outputs.append(output)

        best_output = self.find_best_attack(outputs)
        # we don't need history here actually
        # best_output.history = [deepcopy(o.__dict__) for o in outputs]
        return best_output
