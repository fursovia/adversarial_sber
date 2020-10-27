from typing import List, Optional, Dict, Union

import torch
import numpy as np

from allennlp.data import TextFieldTensors, Vocabulary

from dataclasses import dataclass
from dataclasses_json import dataclass_json

# from advsber.utils.data import decode_indexes
from allennlp.nn.util import get_token_ids_from_text_field_tensors

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader

START_TOKEN = "<START>"
END_TOKEN = "<END>"
MASK_TOKEN = "@@MASK@@"

ModelsInput = Dict[str, Union[TextFieldTensors, torch.Tensor]]


def decode_indexes(
        indexes: torch.Tensor, vocab: Vocabulary, namespace="transactions", drop_start_end: bool = True,
) -> List[str]:
    out = [vocab.get_token_from_index(idx.item(), namespace=namespace) for idx in indexes]

    if drop_start_end:
        return out[1:-1]

    return out


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

    @classmethod
    def from_tensors(cls, inputs: ModelsInput, vocab: Vocabulary) -> "TransactionsData":
        transaction_ids = inputs["transactions"]
        amount_ids = inputs["amounts"]
        label = int(inputs["label"])

        pad_start = np.where(transaction_ids == 0)[0]

        if len(pad_start):
            pad_start = pad_start[0]
            transaction_ids = transaction_ids[:pad_start]
            amount_ids = amount_ids[:pad_start]

        # TODO: also drop paddings
        transactions = decode_indexes(indexes=transaction_ids, vocab=vocab, namespace="transactions", )
        # print(transactions)
        transactions = list(map(int, transactions))

        amounts = decode_indexes(indexes=amount_ids, vocab=vocab, namespace="amounts", )
        # TODO: convert to floats? transform_amounts
        amounts = list(map(int, amounts))

        return cls(transactions=transactions, amounts=amounts, label=label)


def data_to_tensors(
        data: TransactionsData, reader: DatasetReader, vocab: Vocabulary, device: Union[torch.device, int] = -1,
) -> ModelsInput:
    instances = Batch([reader.text_to_instance(**data.to_dict())])

    instances.index_instances(vocab)
    inputs = instances.as_tensor_dict()
    return move_to_device(inputs, device)