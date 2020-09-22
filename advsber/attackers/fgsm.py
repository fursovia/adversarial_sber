from copy import deepcopy

import torch

from advsber.attackers.attacker import Attacker, AttackerOutput
from advsber.settings import TransactionsData


@Attacker.register("fgsm")
class FGSM(Attacker):
    @torch.no_grad()
    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:

        adv_data = deepcopy(data_to_attack)
        output = AttackerOutput(
            data=data_to_attack.to_dict(),
            adversarial_data=adv_data.to_dict(),
            probability=0.5,
            adversarial_probability=0.4,
            prob_diff=0.1,
            wer=5,
        )

        return output
