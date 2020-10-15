from copy import deepcopy

import torch
from torch.distributions import Categorical
from allennlp.models import Model

from advsber.attackers.attacker import Attacker, AttackerOutput
from advsber.settings import TransactionsData
from advsber.utils.data import decode_indexes, data_to_tensors
from advsber.utils.metrics import word_error_rate_on_sequences
from advsber.dataset_readers.transactions_reader import TransactionsDatasetReader


@Attacker.register("sampling_fool")
class SamplingFool(Attacker):
    """
    SamplingFool samples sequences using Masked LM
    """

    def __init__(
        self,
        masked_lm: Model,
        classifier: Model,
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

    def get_lm_logits(self, inputs) -> torch.Tensor:
        logits = self.lm_model(**inputs)["logits"]
        return logits

    @torch.no_grad()
    def attack(self, data_to_attack: TransactionsData) -> AttackerOutput:
        inputs_to_attack = data_to_tensors(data_to_attack, self.reader, self.lm_model.vocab, self.device)

        orig_prob = self.get_clf_probs(inputs_to_attack)[self.label_to_index(data_to_attack.label)].item()

        logits = self.get_lm_logits(inputs_to_attack)
        logits = logits / self.temperature
        probs = torch.softmax(logits, dim=-1)
        probs[:, :, self.special_indexes] = 0.0
        indexes = Categorical(probs=probs[0]).sample((self.num_samples,))
        adversarial_sequences = [decode_indexes(idx, self.lm_model.vocab) for idx in indexes]

        outputs = []
        adv_data = deepcopy(data_to_attack)
        for adv_sequence in adversarial_sequences:
            adv_data.transactions = adv_sequence
            adv_inputs = data_to_tensors(adv_data, self.reader, self.lm_model.vocab, self.device)

            adv_probs = self.get_clf_probs(adv_inputs)
            adv_data.label = self.probs_to_label(adv_probs)
            adv_prob = adv_probs[self.label_to_index(data_to_attack.label)].item()

            output = AttackerOutput(
                data=data_to_attack.to_dict(),
                adversarial_data=adv_data.to_dict(),
                probability=orig_prob,
                adversarial_probability=adv_prob,
                prob_diff=(orig_prob - adv_prob),
                wer=word_error_rate_on_sequences(data_to_attack.transactions, adv_data.transactions),
            )
            outputs.append(output)

        best_output = self.find_best_attack(outputs)
        # we don't need history here actually
        # best_output.history = [deepcopy(o.__dict__) for o in outputs]
        return best_output
