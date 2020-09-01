from allennlp.common.params import Params
import typer
import jsonlines
from tqdm import tqdm

from advsber.attackers import Attacker
from advsber.utils.data import load_jsonlines


def main(config_path: str, samples: int = typer.Option(None, help="Number of samples")):
    params = Params.from_file(config_path)
    attacker = Attacker.from_params(params["attacker"])

    data = load_jsonlines(params["data_path"])[:samples]

    with jsonlines.open(params["output_path"], "w") as writer:
        for el in tqdm(data):
            adversarial_output = attacker.attack(sequence_to_attack=el["text"], label_to_attack=el["label"],)

            writer.write(adversarial_output.__dict__)


if __name__ == "__main__":
    typer.run(main)
