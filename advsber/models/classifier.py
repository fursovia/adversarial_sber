from typing import Dict, Optional

import torch
from torch.nn import Dropout

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from advsber.utils.masker import TokensMasker
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN

@Model.register("BasicClassifier")
class BasicClassifier(Model):
    def __init__(self,
                vocab: Vocabulary,
                transactions_field_embedder: TextFieldEmbedder,
                seq2seq_encoder: Seq2SeqEncoder,
                amounts_field_embedder: Optional[TextFieldEmbedder] = None,
                tokens_masker: Optional[TokensMasker] = None,
                num_classes: int = None
                ) -> None:

        super().__init__(vocab)
        self._transactions_field_embedder = transactions_field_embedder
        self._amounts_field_embedder = amounts_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self.fc = torch.nn.Linear(seq2seq_encoder.get_input_dim(), num_classes)
        self.accuracy = CategoricalAccuracy()
        ignore_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN)
        self.loss_ = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dropout = Dropout(0.1)
        self._tokens_masker = tokens_masker

    def forward(self,
                transactions: TextFieldTensors,
                label: torch.Tensor,
                amounts: Optional[TextFieldTensors] = None,
                **kwargs,
    ) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(transactions)

        if self._tokens_masker is not None:
            transactions, targets = self._tokens_masker.mask_tokens(transactions)

        transaction_embeddings = self._transactions_field_embedder(transactions)

        if amounts is not None and self._amounts_field_embedder is not None:
            amount_embeddings = self._amounts_field_embedder(amounts)
            transaction_embeddings = torch.cat((transaction_embeddings, amount_embeddings), dim=-1)

        contextual_embeddings = self._seq2seq_encoder(transaction_embeddings, mask=mask)
        contextual_embeddings = torch.mean(contextual_embeddings, dim=1)
        contextual_embeddings = self.dropout(contextual_embeddings)
        logits = self.fc(contextual_embeddings)
        probs = torch.nn.functional.softmax(logits)
        loss = self.loss_(logits, label)
        self.accuracy(logits, label)
        output = {'loss': loss, 'probs': probs}
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
