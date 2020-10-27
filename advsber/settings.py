from typing import List, Optional, Dict, Union

import torch
from allennlp.data import TextFieldTensors
from dataclasses import dataclass
from dataclasses_json import dataclass_json


START_TOKEN = "<START>"
END_TOKEN = "<END>"
MASK_TOKEN = "@@MASK@@"

ModelsInput = Dict[str, Union[TextFieldTensors, torch.Tensor]]


@dataclass_json
@dataclass
class TransactionsData:
    transactions: List[int]
    amounts: List[float]
    label: int
    client_id: Optional[int] = None

    def __post_init__(self) -> None:
        assert len(self.transactions) == len(self.amounts)

    def __len__(self) -> int:
        return len(self.transactions)

    # @classmethod
    # def from_tensors(cls, inputs: ModelsInput, vocab: Vocabulary) -> "TransactionsData":
    #
    #     transaction_ids = get_token_ids_from_text_field_tensors(inputs["transactions"])[0]
    #
    #     amount_ids = get_token_ids_from_text_field_tensors(inputs["amounts"])[0]
    #
    #     label = inputs["label"][0].item()
    #
    #     # TODO: also drop paddings
    #     transactions = decode_indexes(indexes=transaction_ids, vocab=vocab, namespace="transactions",)
    #     transactions = list(map(int, transactions))
    #
    #     amounts = decode_indexes(indexes=amount_ids, vocab=vocab, namespace="amounts",)
    #     # TODO: convert to floats? transform_amounts
    #     amounts = list(map(int, amounts))
    #
    #     return cls(transactions=transactions, amounts=amounts, label=label)
