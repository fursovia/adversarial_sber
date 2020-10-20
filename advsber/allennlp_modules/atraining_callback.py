from typing import List, Dict, Any

from allennlp.training.trainer import BatchCallback, GradientDescentTrainer
from allennlp.data.dataloader import TensorDict

from advsber.attackers.attacker import Attacker


@BatchCallback.register("adversarial_training")
class AdversarialTrainingCallback(BatchCallback):
    def __init__(self, attacker_params: str):
        super().__init__()
        self.attacker_params = attacker_params

    def __call__(
        self,
        trainer: GradientDescentTrainer,
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_master: bool,
    ) -> None:

        if is_training:
            attacker = Attacker.from_params(self.attacker_params)
            for batch in batch_inputs:
                adv_batch = attacker.attack(batch)
                batch_outputs = trainer.batch_outputs(adv_batch, for_training=True)
                loss = batch_outputs.get("loss")
                _ = batch_outputs.get("reg_loss")
                loss.backward()
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
