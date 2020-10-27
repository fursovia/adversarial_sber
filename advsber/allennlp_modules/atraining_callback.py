from typing import List, Dict, Any

from allennlp.training.trainer import BatchCallback, GradientDescentTrainer
from allennlp.data.dataloader import TensorDict
from allennlp.data import Batch, DatasetReader
from allennlp.common import Params

#from advsber.attackers.attacker import Attacker
from advsber.attackers.fgsm import FGSM
from advsber.settings import TransactionsData
from tqdm import tqdm


@BatchCallback.register("adversarial_training")
class AdversarialTrainingCallback(BatchCallback):
    def __init__(self, attacker_params: Dict[str, Any], reader: DatasetReader):
        super().__init__()
        self.attacker_params = Params(attacker_params)
        print(self.attacker_params)
        reader = DatasetReader.from_params(self.attacker_params["reader"])
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

        batch_inputs = batch_inputs[0]
        labels = batch_inputs['label'].numpy()


        if is_training:

            attacker = FGSM(classifier=trainer.model, reader=self.reader, device=0)

            instances = []
            for index in tqdm(range(len(labels))):

                batch = {'transactions': batch_inputs['transactions']['tokens']['tokens'][index].numpy(),
                         'amounts': batch_inputs['amounts']['tokens']['tokens'][index].numpy(),
                         'label': int(labels[index]),
                         'client_id': int(batch_inputs['client_id'].numpy()[index])}

                data = TransactionsData.from_tensors(inputs=batch, vocab=trainer.model.vocab)

                adv_data = attacker.attack(data)
                adv_data = adv_data.to_dict()

                adv_data = {'transactions': adv_data['adversarial_data']['transactions'],
                            'amounts': adv_data['adversarial_data']['amounts'],
                            'label': adv_data['data']['label'],
                            'client_id': adv_data['data']['client_id']}

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