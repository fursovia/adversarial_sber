from typing import Dict, Optional

import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.fields import ArrayField
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp_models.lm.modules import LinearLanguageModelHead

from advsber.allennlp_modules.metrics import FixedPerplexity
from advsber.utils.masker import TokensMasker


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
        #self._amounts_field_embedder = amounts_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._head = LinearLanguageModelHead(
            vocab=vocab, input_dim=self._seq2seq_encoder.get_output_dim(), vocab_namespace="transactions"
        )
        self._tokens_masker = tokens_masker

        ignore_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN)
        self._loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self._perplexity = FixedPerplexity()

    def forward(
        self, transactions: TextFieldTensors, amounts: Optional[ArrayField] = None, **kwargs,
    ) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(transactions)

        if self._tokens_masker is not None:
            transactions, targets = self._tokens_masker.mask_tokens(transactions)
        else:
            targets = transactions

        transaction_embeddings = self._transactions_field_embedder(transactions)
        if amounts is not None: #and self._amounts_field_embedder is not None:
            #amount_embeddings = self._amounts_field_embedder(amounts)
            transaction_embeddings = torch.cat((transaction_embeddings, amounts.unsqueeze(-1)), dim=-1)

        contextual_embeddings = self._seq2seq_encoder(transaction_embeddings, mask)

        # take PAD tokens into account when decoding
        logits = self._head(contextual_embeddings)

        output_dict = dict(contextual_embeddings=contextual_embeddings, logits=logits, mask = mask)

        output_dict["loss"] = self._loss(
            logits.transpose(1, 2),
            # TODO: it is not always tokens-tokens
            targets["tokens"]["tokens"],
        )
        self._perplexity(output_dict["loss"])
        return output_dict

    def get_metrics(self, reset: bool = False):
        return {"perplexity": self._perplexity.get_metric(reset=reset)}
