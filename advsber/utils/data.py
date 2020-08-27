from typing import Union, List, Dict, Any

import torch
from allennlp.data import TextFieldTensors, Batch
from allennlp.nn.util import move_to_device
import jsonlines

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary


START_TOKEN = "<START>"
END_TOKEN = "<END>"
MASK_TOKEN = "@@MASK@@"


def sequence_to_tensors(
    sequence: str, reader: DatasetReader, vocab: Vocabulary, device: Union[torch.device, int],
) -> TextFieldTensors:
    instances = Batch([reader.text_to_instance(sequence)])

    instances.index_instances(vocab)
    inputs = instances.as_tensor_dict()["tokens"]
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
