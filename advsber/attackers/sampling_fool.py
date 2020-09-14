from copy import deepcopy

import torch
from torch.distributions import Categorical

from advsber.attackers.attacker import Attacker, AttackerOutput
from advsber.settings import TransactionsData
from advsber.utils.data import decode_indexes, data_to_tensors
from advsber.utils.metrics import word_error_rate_on_sequences
from advsber.models.masked_lm import MaskedLanguageModel
from advsber.dataset_readers.transactions_reader import TransactionsDatasetReader
from advsber.models.classifier import TransactionsClassifier


@Attacker.register("sampling_fool")
class SamplingFool(Attacker):
    """
    SamplingFool samples sequences using Masked LM
    """

    def __init__(
        self,
        masked_lm: MaskedLanguageModel,
        classifier: TransactionsClassifier,
        reader: TransactionsDatasetReader,
        num_samples: int = 100,
        temperature: float = 1.0,
        device: int = -1,
    ) -> None:
        super().__init__(classifier=classifier, reader=reader, device=device)
        self.lm_model = masked_lm
        # disable masker by hands
        self.lm_model._tokens_masker = None
        self.lm_model.eval()

        if self.device >= 0 and torch.cuda.is_available():
            self.lm_model.cuda(self.device)

        self.num_samples = num_samples
        self.temperature = temperature

    # TODO: add typing
    def get_clf_probs(self, inputs) -> torch.Tensor:
        probs = self.classifier(inputs)["probs"][0]
        return probs

    def get_lm_logits(self, inputs) -> torch.Tensor:
        logits = self.lm_model(inputs)["logits"]
        return logits

    @torch.no_grad()
    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:
        inputs_to_attack = data_to_tensors(data_to_attack, self.reader, self.lm_model.vocab, self.device)

        orig_prob = self.get_clf_probs(inputs_to_attack)[data_to_attack.label].item()

        logits = self.get_lm_logits(inputs_to_attack)
        indexes = Categorical(logits=logits[0] / self.temperature).sample((self.num_samples,))
        adversarial_sequences = [decode_indexes(idx, self.lm_model.vocab) for idx in indexes]

        outputs = []
        adv_data = deepcopy(data_to_attack)
        for adv_sequence in adversarial_sequences:
            adv_data.transactions = adv_sequence
            adv_inputs = data_to_tensors(adv_data, self.reader, self.lm_model.vocab, self.device)

            adv_probs = self.calculate_probs(adv_inputs)
            adv_prob = adv_probs[data_to_attack.label].item()

            output = AttackerOutput(
                data=data_to_attack,
                adversarial_data=adv_data,
                probability=orig_prob,
                adversarial_probability=adv_prob,
                prob_diff=(orig_prob - adv_prob),
                wer=word_error_rate_on_sequences(data_to_attack.transactions, adv_data.transactions)
            )
            outputs.append(output)

        best_output = self.find_best_attack(outputs)
        # we don't need history here actually
        # best_output.history = [deepcopy(o.__dict__) for o in outputs]
        return best_output
