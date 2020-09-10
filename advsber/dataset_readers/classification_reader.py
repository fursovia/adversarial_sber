import allennlp
import tempfile
import json

import torch.nn as nn
from typing import Dict, Iterable, List, Tuple, Union, Optional

from allennlp.data import TextFieldTensors

from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, Seq2SeqEncoder
from allennlp.nn import util

from allennlp.data import DataLoader, DatasetReader, Instance, TextFieldTensors, Vocabulary
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.training.util import evaluate
from allennlp.common.file_utils import cached_path

@DatasetReader.register("classification-csv")
class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, text: str, label: int = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields = {'text': text_field}
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path), "r") as data_file:
            iter_ = 0
            for line in data_file.readlines():
                if iter_ > 0:
                    text, label = line.strip().split(',')
                    yield self.text_to_instance(text, label)
                else:
                    iter_ = 1