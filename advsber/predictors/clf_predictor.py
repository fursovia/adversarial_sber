from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("transactions")
class TransactionsPredictor(Predictor):
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        transactions = json_dict["transactions"]
        amounts = json_dict["amounts"]
        return self._dataset_reader.text_to_instance(transactions=transactions, amounts=amounts)
