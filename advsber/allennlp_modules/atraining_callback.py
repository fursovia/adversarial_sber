from typing import List, Dict, Any

from allennlp.training.trainer import BatchCallback, GradientDescentTrainer
from allennlp.data.dataloader import TensorDict
from allennlp.data import Batch, DatasetReader

from advsber.attackers.attacker import Attacker
from advsber.settings import TransactionsData


@BatchCallback.register("adversarial_training")
class AdversarialTrainingCallback(BatchCallback):
    def __init__(self, attacker_params: str, reader: DatasetReader):
        super().__init__()
        self.attacker_params = attacker_params
        self.reader = reader

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

                instances = []
                for element in batch:
                    data = TransactionsData.from_tensors(inputs=element, vocab=trainer.model.vocab)
                    adv_data = attacker.attack(data)

                    instance = self.reader.text_to_instance(**adv_data)
                    instances.append(instance)

                new_batch = Batch(instances)
                new_batch.index_instances(vocab=trainer.model.vocab)

                new_batch = new_batch.as_tensor_dict()

                batch_outputs = trainer.batch_outputs(new_batch, for_training=True)
                loss = batch_outputs.get("loss")
                _ = batch_outputs.get("reg_loss")
                loss.backward()
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()
