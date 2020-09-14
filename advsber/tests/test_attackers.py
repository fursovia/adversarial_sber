from pathlib import Path

from allennlp.common import Params

import advsber
from advsber.utils.data import load_jsonlines
from advsber.settings import TransactionsData


PROJECT_ROOT = (Path(__file__).parent / ".." / "..").resolve()
CONFIG_DIR = PROJECT_ROOT / "configs"


class TestTransactionAttackers:

    def test_from_params(self):
        data = load_jsonlines(str(PROJECT_ROOT / "presets/age/sample.jsonl"))

        for config_path in (CONFIG_DIR / "attackers").glob("*.jsonnet"):
            try:
                params = Params.from_file(
                    str(config_path), ext_vars={
                        "DATA_PATH": "",
                        "OUTPUT_PATH": "",
                        "CLF_PATH": str(PROJECT_ROOT / "presets/age/clf.model.tar.gz"),
                        "MASKED_LM_PATH": str(PROJECT_ROOT / "presets/age/lm.model.tar.gz")
                    }
                )
                attacker = advsber.Attacker.from_params(params["attacker"])
            except Exception as e:
                raise AssertionError(f"unable to load params from {config_path}, because {e}")

            output = attacker.attack(TransactionsData(**data[0]))
            assert isinstance(output, advsber.AttackerOutput)
