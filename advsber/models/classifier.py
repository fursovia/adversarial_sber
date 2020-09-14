from typing import Dict, Optional

import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from allennlp.nn import util


@Model.register("transactions_classifier")
class TransactionsClassifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        transactions_field_embedder: TextFieldEmbedder,
        seq2vec_encoder: Seq2VecEncoder,
        seq2seq_encoder: Optional[Seq2SeqEncoder] = None,
        amounts_field_embedder: Optional[TextFieldEmbedder] = None,
        num_labels: Optional[int] = None,
    ) -> None:

        super().__init__(vocab)
        self._transactions_field_embedder = transactions_field_embedder
        self._seq2vec_encoder = seq2vec_encoder
        self._seq2seq_encoder = seq2seq_encoder
        self._amounts_field_embedder = amounts_field_embedder

        num_labels = num_labels or vocab.get_vocab_size("labels")
        self._classification_layer = torch.nn.Linear(self._seq2vec_encoder.get_output_dim(), num_labels)

        self._loss = torch.nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()

    def forward(
            self,
            transactions: TextFieldTensors,
            label: Optional[torch.Tensor] = None,
            amounts: Optional[TextFieldTensors] = None,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(transactions)
        transaction_embeddings = self._transactions_field_embedder(transactions)

        if amounts is not None and self._amounts_field_embedder is not None:
            amount_embeddings = self._amounts_field_embedder(amounts)
            transaction_embeddings = torch.cat((transaction_embeddings, amount_embeddings), dim=-1)

        if self._seq2seq_encoder is not None:
            transaction_embeddings = self._seq2seq_encoder(transaction_embeddings, mask=mask)

        embedded_transactions = self._seq2vec_encoder(transaction_embeddings, mask=mask)

        logits = self._classification_layer(embedded_transactions)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = dict(
            logits=logits,
            probs=probs,
            token_ids=util.get_token_ids_from_text_field_tensors(transactions)
        )
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics
