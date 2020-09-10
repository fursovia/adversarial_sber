import allennlp
import tempfile
import jsonlines

import torch.nn as nn
from typing import Dict, Iterable, List


from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.common.file_utils import cached_path


@DatasetReader.register("ClassificationReader")
class ClassificationReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, transactions=None, sums=None, label=None, client_id=None) -> Instance:
        tokens_transactions = self.tokenizer.tokenize(transactions)
        tokens_sums = self.tokenizer.tokenize(sums)
        
        if self.max_tokens:
            tokens_transactions = tokens_transactions[:self.max_tokens]
            tokens_sums = tokens_sums[:self.max_tokens]
            
        transactions = TextField(tokens_transactions, self.token_indexers)
        sums = TextField(tokens_sums, self.token_indexers)
        
        fields = {'transactions': transactions}
        fields['amounts'] = sums
        fields['label'] = LabelField(label)
        fields['client_id'] = client_id
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with jsonlines.open(cached_path(file_path), "r") as reader:
            for line in reader:
                transactions = line['transactions']
                transactions = ' '.join(transactions)
                sums = line['sums']
                sums = ' '.join(map(str, sums)) 
                label = line['label']
                client_id = line['client_id']
                yield self.text_to_instance(transactions, sums, str(label), client_id)
