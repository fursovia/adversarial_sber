from typing import Dict, Optional

import torch
from torch.nn import Dropout
from torch.nn.functional import relu

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from advsber.utils.masker import TokensMasker
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN


#Realization of Constrastive Loss for Metric Learning Model
def ContrastiveLoss(embedding, labels):
    labels = labels.unsqueeze(0)
    mask = (labels - labels.T) == 0
    mask = mask.double()

    n = embedding.size(0)
    m = embedding.size(0)
    d = embedding.size(1)

    x = embedding.unsqueeze(1).expand(n, m, d)
    y = embedding.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, 2).sum(2)
    part_1 = mask * dist
    part_2 = (torch.ones_like(dist) - mask) * relu(1e-3 * torch.ones_like(dist) - torch.sqrt(dist)).pow(2)
    L = part_1.sum() + part_2.sum()
    return L

@Model.register("metric_learning_2")
class metriclr_2(Model):
    def __init__(self,
                vocab: Vocabulary,
                transactions_field_embedder: TextFieldEmbedder,
                seq2seq_encoder_transactions: Seq2SeqEncoder,
                seq2seq_encoder_amounts: Seq2SeqEncoder,
                amounts_field_embedder: Optional[TextFieldEmbedder] = None,
                tokens_masker: Optional[TokensMasker] = None,
                num_classes: int = None,
                alpha : float = None,
                beta: float = None
                ) -> None:

        super().__init__(vocab)
        self._transactions_field_embedder = transactions_field_embedder
        self._amounts_field_embedder = amounts_field_embedder
        self._seq2seq_encoder_transactions = seq2seq_encoder_transactions
        self._seq2seq_encoder_amounts = seq2seq_encoder_amounts
        self.fc = torch.nn.Linear(2*seq2seq_encoder_transactions.get_input_dim(), num_classes)
        self.accuracy = CategoricalAccuracy()
        ignore_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN)
        self.loss_ = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dropout = Dropout(0.1)
        self._tokens_masker = tokens_masker
        self.alpha = alpha
        self.beta = beta

    def forward(self,
                transactions: TextFieldTensors,
                label: torch.Tensor,
                amounts: Optional[TextFieldTensors] = None,
                client_id: int = None,
                **kwargs,
    ) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(transactions)

        if self._tokens_masker is not None:
            transactions, targets = self._tokens_masker.mask_tokens(transactions)

        transaction_embeddings = self._transactions_field_embedder(transactions)

        if amounts is not None and self._amounts_field_embedder is not None:
            amount_embeddings = self._amounts_field_embedder(amounts)

        transaction_embeddings = self._seq2seq_encoder_transactions(transaction_embeddings, mask=mask)
        amount_embeddings = self._seq2seq_encoder_transactions(amount_embeddings)

        transaction_embeddings = torch.mean(transaction_embeddings, dim=1)
        amount_embeddings = torch.mean(amount_embeddings, dim=1)

        loss_1 = ContrastiveLoss(transaction_embeddings, client_id)
        loss_2 = ContrastiveLoss(amount_embeddings, client_id)

        contextual_embeddings = torch.cat((transaction_embeddings, amount_embeddings), dim=1)
        contextual_embeddings = self.dropout(contextual_embeddings)
        logits = self.fc(contextual_embeddings)
        probs = torch.nn.functional.softmax(logits)
        loss_3 = self.loss_(logits, label)
        loss = self.alpha*loss_1 + self.alpha*loss_2 + self.beta*loss_3
        self.accuracy(logits, label)
        output = {'loss': loss, 'probs': probs}
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
