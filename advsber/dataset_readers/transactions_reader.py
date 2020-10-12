from typing import List, Optional
import jsonlines
import math
import logging

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer, Token
from allennlp.models.archival import load_archive

from advsber.utils.data import load_discretizer, transform_amounts
from advsber.settings import START_TOKEN, END_TOKEN

logger = logging.getLogger(__name__)


@DatasetReader.register("transactions_reader")
class TransactionsDatasetReader(DatasetReader):
    def __init__(self, discretizer_path: str, max_sequence_length: int = None, lazy: bool = False,) -> None:
        super().__init__(lazy=lazy)

        self.discretizer = load_discretizer(discretizer_path)
        self._max_sequence_length = max_sequence_length or math.inf
        self._tokenizer = WhitespaceTokenizer()
        self._start_token = Token(START_TOKEN)
        self._end_token = Token(END_TOKEN)

    def _add_start_end_tokens(self, tokens: List[Token]) -> List[Token]:
        return [self._start_token] + tokens + [self._end_token]

    def text_to_instance(
        self,
        transactions: List[int],
        amounts: List[float],
        label: Optional[int] = None,
        client_id: Optional[int] = None,
    ) -> Instance:

        transactions = " ".join(map(str, transactions))
        amounts = transform_amounts(amounts, self.discretizer)
        amounts = " ".join(amounts)

        transactions = self._tokenizer.tokenize(transactions)
        amounts = self._tokenizer.tokenize(amounts)

        transactions = self._add_start_end_tokens(transactions)
        amounts = self._add_start_end_tokens(amounts)

        fields = {
            "transactions": TextField(transactions, {"tokens": SingleIdTokenIndexer("transactions")}),
            "amounts": TextField(amounts, {"tokens": SingleIdTokenIndexer("amounts")}),
        }

        if label is not None:
            fields["label"] = LabelField(label=str(label), skip_indexing=False)

        if client_id is not None:
            fields["client_id"] = LabelField(label=client_id, skip_indexing=True, label_namespace="client_id")

        return Instance(fields)

    def _read(self, file_path: str):

        logger.info("Loading data from %s", file_path)
        dropped_instances = 0

        with jsonlines.open(cached_path(file_path), "r") as reader:
            for items in reader:
                transactions = items["transactions"]
                amounts = items["amounts"]
                assert len(transactions) == len(amounts)

                instance = self.text_to_instance(
                    transactions=transactions,
                    amounts=amounts,
                    label=items.get("label"),
                    client_id=items.get("client_id"),
                )
                if instance.fields["transactions"].sequence_length() <= self._max_sequence_length:
                    yield instance
                else:
                    dropped_instances += 1

        if not dropped_instances:
            logger.info(f"No instances dropped from {file_path}.")
        else:
            logger.warning(f"Dropped {dropped_instances} instances from {file_path}.")

    @classmethod
    def from_archive(cls, archive_file: str) -> "TransactionsDatasetReader":
        config = load_archive(archive_file).config["dataset_reader"]
        assert config.pop("type") == "transactions_reader"
        return cls(**config)


TransactionsDatasetReader.register("from_archive", constructor="from_archive")(TransactionsDatasetReader)