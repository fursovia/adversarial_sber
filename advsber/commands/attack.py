from allennlp.common.params import Params
import typer
import jsonlines
from tqdm import tqdm

from advsber.attackers import Attacker
from advsber.utils.data import load_jsonlines


def attack(config_path: str):
    params = Params.from_file(config_path)
    attacker = Attacker.from_params(params["attacker"])

    data = load_jsonlines(params["data_path"])

    with jsonlines.open(params["output_path"], "w") as writer:
        for el in tqdm(data):
            adversarial_output = attacker.attack(
                sequence_to_attack=el["sequence"],
                label_to_attack=el["label"],
            )

            writer.write(adversarial_output.__dict__)


if __name__ == "__main__":
    typer.run(attack)
