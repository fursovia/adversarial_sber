from typing import Union, List, Dict, Any, Sequence
from itertools import chain

import torch
import pickle
import jsonlines
import numpy as np
from allennlp.data import TextFieldTensors, Batch
from allennlp.nn.util import move_to_device
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from sklearn.preprocessing import KBinsDiscretizer


START_TOKEN = "<START>"
END_TOKEN = "<END>"
MASK_TOKEN = "@@MASK@@"


def sequence_to_tensors(
    sequence: str, reader: DatasetReader, vocab: Vocabulary, device: Union[torch.device, int] = -1,
) -> TextFieldTensors:
    instances = Batch([reader.text_to_instance(sequence)])

    instances.index_instances(vocab)
    inputs = instances.as_tensor_dict()["tokens"]
    return move_to_device(inputs, device)


def data_to_tensors(
    data: Dict[str, Any], reader: DatasetReader, vocab: Vocabulary, device: Union[torch.device, int] = -1
):
    instances = Batch([reader.text_to_instance(**data)])

    instances.index_instances(vocab)
    inputs = instances.as_tensor_dict()
    return move_to_device(inputs, device)


def decode_indexes(indexes: torch.Tensor, vocab: Vocabulary) -> str:
    out = [vocab.get_token_from_index(idx.item()) for idx in indexes]
    out = [o for o in out if o not in [START_TOKEN, END_TOKEN]]
    return " ".join(out)


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
    print(amounts.shape)
    amounts = discretizer.transform([[x] for x in amounts])
    # unpack and covert float -> int -> str
    amounts = list(map(str, (map(int, chain(*amounts)))))
    return amounts
