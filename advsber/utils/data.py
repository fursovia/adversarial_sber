from typing import Union, List, Dict, Any, Sequence
from itertools import chain

import torch
import pickle
import jsonlines
import numpy as np
from allennlp.data import Batch
from allennlp.nn.util import move_to_device
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from sklearn.preprocessing import KBinsDiscretizer

from advsber.settings import START_TOKEN, END_TOKEN, TransactionsData


def data_to_tensors(
    data: TransactionsData, reader: DatasetReader, vocab: Vocabulary, device: Union[torch.device, int] = -1
):

    instances = Batch([reader.text_to_instance(**data.to_dict())])

    instances.index_instances(vocab)
    inputs = instances.as_tensor_dict()
    return move_to_device(inputs, device)


def decode_indexes(
        indexes: torch.Tensor,
        vocab: Vocabulary,
        namespace="transactions",
        drop_start_end: bool = True,
) -> List[str]:
    out = [vocab.get_token_from_index(idx.item(), namespace=namespace) for idx in indexes]

    if drop_start_end:
        return out[1:-1]

    return out


def load_jsonlines(path: str) -> List[Dict[str, Any]]:
    data = []
    with jsonlines.open(path, "r") as reader:
        for items in reader:
            data.append(items)
    return data


def write_jsonlines(data: Sequence[Dict[str, Any]], path: str) -> None:
    with jsonlines.open(path, "w") as writer:
        for ex in data:
            writer.write(ex)


def generate_transaction_amounts(total_amount: float, num_transactions: int, alpha: float = 1.0) -> List[float]:
    assert total_amount > 0
    values = np.random.dirichlet(np.ones(num_transactions) * alpha, size=1) * total_amount
    values = values.tolist()[0]
    return values


def load_discretizer(discretizer_path: str) -> KBinsDiscretizer:
    with open(discretizer_path, "rb") as f:
        discretizer: KBinsDiscretizer = pickle.load(f)
        assert discretizer.encode == "ordinal"

    return discretizer


def transform_amounts(amounts: List[float], discretizer: KBinsDiscretizer) -> List[str]:
    amounts = discretizer.transform([[x] for x in amounts])
    # unpack and covert float -> int -> str
    amounts = list(map(str, (map(int, chain(*amounts)))))
    return amounts
