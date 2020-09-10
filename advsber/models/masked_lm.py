from typing import Dict, Optional

import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
# from allennlp.training.metrics import Perplexity
from allennlp.nn.util import get_text_field_mask
from allennlp_models.lm.modules import LinearLanguageModelHead

from advsber.utils.masker import TokensMasker


from allennlp.training.metrics.average import Average
from allennlp.training.metrics.metric import Metric


@Metric.register("perplexity")
class Perplexity(Average):
    """
    Perplexity is a common metric used for evaluating how well a language model
    predicts a sample.

    Notes
    -----
    Assumes negative log likelihood loss of each batch (base e). Provides the
    average perplexity of the batches.
    """

    def get_metric(self, reset: bool = False):
        """
        # Returns

        The accumulated perplexity.
        """
        average_loss = super().get_metric(reset)
        if average_loss == 0:
            perplexity = 0.0

        # Exponentiate the loss to compute perplexity
        perplexity = float(torch.exp(torch.tensor(average_loss)))

        return perplexity



@Model.register("masked_lm")
class MaskedLanguageModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        transactions_field_embedder: TextFieldEmbedder,
        seq2seq_encoder: Seq2SeqEncoder,
        amounts_field_embedder: Optional[TextFieldEmbedder] = None,
        tokens_masker: Optional[TokensMasker] = None,
    ) -> None:
        super().__init__(vocab)
        self._transactions_field_embedder = transactions_field_embedder
        self._amounts_field_embedder = amounts_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._head = LinearLanguageModelHead(
            vocab=vocab, input_dim=self._seq2seq_encoder.get_output_dim(), vocab_namespace="transactions"
        )
        self._tokens_masker = tokens_masker

        ignore_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN)
        self._loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self._perplexity = Perplexity()

    def forward(
        self,
        transactions: TextFieldTensors,
        amounts: Optional[TextFieldTensors] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(transactions)

        if self._tokens_masker is not None:
            transactions, targets = self._tokens_masker.mask_tokens(transactions)
        else:
            targets = transactions

        transaction_embeddings = self._transactions_field_embedder(transactions)
        if amounts is not None and self._amounts_field_embedder is not None:
            amount_embeddings = self._amounts_field_embedder(amounts)
            transaction_embeddings = torch.cat((transaction_embeddings, amount_embeddings), dim=-1)

        contextual_embeddings = self._seq2seq_encoder(transaction_embeddings, mask)

        # take PAD tokens into account when decoding
        logits = self._head(contextual_embeddings)

        output_dict = dict(contextual_embeddings=contextual_embeddings, logits=logits, mask=mask)

        output_dict["loss"] = self._loss(
            logits.transpose(1, 2),
            # TODO: it is not always tokens-tokens
            targets["tokens"]["tokens"],
        )
        self._perplexity(output_dict["loss"])
        return output_dict

    def get_metrics(self, reset: bool = False):
        return {"perplexity": self._perplexity.get_metric(reset=reset)}
