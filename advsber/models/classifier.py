from typing import Dict, Optional

import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp_models.lm.modules import LinearLanguageModelHead

@Model.register("BasicClassifier")
class BasicClassifier(Model):
    def __init__(self,
                vocab: Vocabulary,
                transactions_field_embedder: TextFieldEmbedder,
                seq2seq_encoder: Seq2SeqEncoder,
                amounts_field_embedder: Optional[TextFieldEmbedder] = None,
                ) -> None:

        super().__init__(vocab)
        self._transactions_field_embedder = transactions_field_embedder
        self._amounts_field_embedder = amounts_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        num_labels = vocab.get_vocab_size("labels")
        self.fc = torch.nn.Linear(256, num_labels)

        #self._head = LinearLanguageModelHead(vocab=vocab,
        #                                     input_dim=self._seq2seq_encoder.get_output_dim(),
        #                                     vocab_namespace="transactions")

        self._loss = torch.nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()

    def forward(self,
                transactions: TextFieldTensors,
                label: torch.Tensor,
                amounts: Optional[TextFieldTensors] = None,
                **kwargs,
    ) -> Dict[str, torch.Tensor]:
        transaction_embeddings = self._transactions_field_embedder(transactions)
        print(transaction_embeddings.shape)
        if amounts is not None and self._amounts_field_embedder is not None:
            amount_embeddings = self._amounts_field_embedder(amounts)
            transaction_embeddings = torch.cat((transaction_embeddings, amount_embeddings), dim=-1)
        shapes = transaction_embeddings.shape
        #transaction_embeddings = transaction_embeddings.reshape(shapes[1], shapes[0], shapes[2])
        print(transaction_embeddings.shape)
        contextual_embeddings = self._seq2seq_encoder(transaction_embeddings, mask=None)
        contextual_embeddings = torch.mean(contextual_embeddings, dim=1)
        logits = self.fc(contextual_embeddings)
        #logits = self._head(contextual_embeddings)
        print(logits.shape, 'AAAAAAAAAA')
        probs = torch.nn.functional.softmax(logits)
        loss = torch.nn.functional.cross_entropy(logits, label)
        self._accuracy(logits, label)
        output = {'loss': loss, 'probs': probs}
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
