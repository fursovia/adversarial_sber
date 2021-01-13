from typing import Dict, Optional

import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import Auc
from allennlp.nn.util import get_text_field_mask
from allennlp.nn import util

@Model.register("rtd_encoder")
class rtd_encoder(Model):
    default_predictor = "transactions"

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
        self._classification_layer = torch.nn.Linear(self._seq2vec_encoder.get_output_dim(), 300)

        self._loss = torch.nn.CrossEntropyLoss()
        self._accuracy = CategoricalAccuracy()

    def get_transaction_embeddings(self, transactions: TextFieldTensors) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(transactions)
        transaction_embeddings = self._transactions_field_embedder(transactions)
        return {"mask": mask, "transaction_embeddings": transaction_embeddings}

    def forward_on_transaction_embeddings(
        self,
        transaction_embeddings: torch.Tensor,
        mask: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        amounts: Optional[TextFieldTensors] = None,
    ) -> Dict[str, torch.Tensor]:

        if amounts is not None and self._amounts_field_embedder is not None:
            amount_embeddings = self._amounts_field_embedder(amounts)
            transaction_embeddings = torch.cat((transaction_embeddings, amount_embeddings), dim=-1)

        if self._seq2seq_encoder is not None:
            transaction_embeddings = self._seq2seq_encoder(transaction_embeddings, mask=mask)

        embedded_transactions = self._seq2vec_encoder(transaction_embeddings, mask=mask)
        
        logits = self._classification_layer(embedded_transactions)
        logits = embedded_transactions.reshape(embedded_transactions.shape[0]*embedded_transactions.shape[1]//2, 2)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = dict(logits=logits, probs=probs)
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict

    def forward(
        self,
        transactions: TextFieldTensors,
        label: Optional[torch.Tensor] = None,
        amounts: Optional[TextFieldTensors] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
              
        
        mask = torch.zeros((transactions['tokens']['tokens'].shape[0],self._seq2vec_encoder.get_output_dim()//2) , dtype=torch.int64)
        
        zeros_cat = torch.zeros((transactions['tokens']['tokens'].shape[0],self._seq2vec_encoder.get_output_dim()//2 - transactions['tokens']['tokens'].shape[1]), dtype=torch.int64, device=transactions['tokens']['tokens'].device)
        
        transactions['tokens']['tokens'] = torch.cat((transactions['tokens']['tokens'], zeros_cat), 1)
        amounts['tokens']['tokens'] = torch.cat((amounts['tokens']['tokens'], zeros_cat), 1)
        
        transactions_val = transactions['tokens']['tokens'].detach().cpu().numpy()
        lengths = [(x!=0).nonzero()[-1][-1]+1 for x in transactions_val]
        
        for i, length in enumerate(lengths):
            mask[i, :length] = 1

        replace_prob = 0.3
        to_replace = torch.bernoulli(mask * replace_prob).bool()
        
        sampled_trx_ids = torch.multinomial(
            mask.flatten().float(),
            num_samples=to_replace.sum().item(),
            replacement=True
        )
        
        labels = to_replace.long().flatten()
        labels = torch.tensor(labels, dtype=torch.long, device=label.device)

        transactions_replace = transactions_val.reshape(-1)[sampled_trx_ids]
        amounts_replace = amounts['tokens']['tokens'].reshape(-1)[sampled_trx_ids]
        
        transactions_replace = torch.tensor(transactions_replace, dtype=torch.long, device=label.device)
        transactions['tokens']['tokens'][to_replace] = transactions_replace
        amounts['tokens']['tokens'][to_replace] = amounts_replace
        
        emb_out = self.get_transaction_embeddings(transactions)
        
        output_dict = self.forward_on_transaction_embeddings(
            transaction_embeddings=emb_out["transaction_embeddings"],
            mask=emb_out["mask"],
            label=labels,
            amounts=amounts,
        )

        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(transactions)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"Accuracy": self._accuracy.get_metric(reset)}
        return metrics

    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary("labels").get(label_idx, str(label_idx))
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict
