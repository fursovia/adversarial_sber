from allennlp.common import Params
import pytest

import advsber
from advsber.tests import CONFIG_DIR, PRESETS_DIR
from advsber.utils.data import load_jsonlines
from advsber.settings import TransactionsData


ATTACKERS_CONFIG_DIR = CONFIG_DIR / "attackers"
CONFIGS = list(ATTACKERS_CONFIG_DIR.glob("*.jsonnet"))

CLASSIFIERS = list(PRESETS_DIR.glob("*/models/*_clf/*.tar.gz"))


class TestTransactionAttackers:

    age_test_data = load_jsonlines(str(PRESETS_DIR / "age" / "sample.jsonl"))
    gender_test_data = load_jsonlines(str(PRESETS_DIR / "gender" / "sample.jsonl"))

    @pytest.mark.parametrize("clf_path", CLASSIFIERS)
    @pytest.mark.parametrize("config_path", CONFIGS)
    def test_from_params(self, config_path, clf_path):

        dataset = clf_path.parent.parent.parent.name
        if dataset == "age":
            data = self.age_test_data
        elif dataset == "gender":
            data = self.gender_test_data
        else:
            raise NotImplementedError

        try:
            params = Params.from_file(
                str(config_path),
                ext_vars={
                    "DATA_PATH": "",
                    "OUTPUT_PATH": "",
                    "CLF_PATH": str(clf_path),
                    "MASKED_LM_PATH": str(
                        clf_path.parent.parent / "lm/bert_with_amounts.tar.gz"
                    ),
                },
            )
            params["attacker"]["device"] = -1
            attacker = advsber.Attacker.from_params(params["attacker"])
        except Exception as e:
            raise AssertionError(
                f"unable to load params from {config_path}, because {e}"
            )

        output = attacker.attack(TransactionsData(**data[0]))
        assert isinstance(output, advsber.AttackerOutput)
        assert isinstance(output.wer, int)
        assert output.wer >= 0
        assert isinstance(output.prob_diff, float)
        assert abs(output.prob_diff) <= 1.0
        assert isinstance(output.probability, float)
        assert output.probability >= 0.0
        assert isinstance(output.adversarial_probability, float)
        assert output.adversarial_probability >= 0.0
