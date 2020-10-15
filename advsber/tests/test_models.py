import pytest
from allennlp.common import Params
from allennlp.data import Batch, DatasetReader, Vocabulary
from allennlp.models import Model

from advsber.tests import CONFIG_DIR, PRESETS_DIR

CLF_CONFIGS_DIR = CONFIG_DIR / "classifiers"
CONFIGS = list(CLF_CONFIGS_DIR.glob("*.jsonnet"))
DATA_PATH = PRESETS_DIR / "age" / "sample.jsonl"
DISCRETIZER_PATH = PRESETS_DIR / "age" / "discretizers" / "100_quantile"
VOCAB_PATH = PRESETS_DIR / "age" / "vocabs" / "100_quantile"


def test_there_is_at_least_one_config():
    assert CONFIGS


class TestModel:
    @pytest.mark.parametrize("config_path", CONFIGS)
    def test_create_models_from_allennlp_configs(self, config_path):

        params = Params.from_file(
            str(config_path),
            ext_vars={
                "CLF_TRAIN_DATA_PATH": "",
                "CLF_VALID_DATA_PATH": "",
                "DISCRETIZER_PATH": str(DISCRETIZER_PATH),
                "VOCAB_PATH": str(VOCAB_PATH),
            },
        )

        reader = DatasetReader.from_params(params["dataset_reader"])

        instances = reader.read(DATA_PATH)
        vocab = Vocabulary.from_instances(instances)
        num_labels = vocab.get_vocab_size(namespace="labels")

        batch = Batch(instances)
        batch.index_instances(vocab)

        try:
            model = Model.from_params(params=params["model"], vocab=vocab)
        except Exception as e:
            raise AssertionError(f"unable to load params from {config_path}") from e

        output_dict = model(**batch.as_tensor_dict())

        assert "probs" in output_dict
        assert len(output_dict["probs"].shape) == 2
        assert output_dict["probs"].shape[0] == len(instances)
        assert output_dict["probs"].shape[1] == num_labels
