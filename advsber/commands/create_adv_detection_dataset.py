from typing import List, Dict, Any

from pathlib import Path
import typer
from sklearn.model_selection import train_test_split

from advsber.utils.data import load_jsonlines, write_jsonlines


def create_dataset_from_output(output: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dataset = []
    for example in output:
        transactions = example["data"]["transactions"]
        amounts = example["data"]["amounts"]
        adv_transactions = example["adversarial_data"]["transactions"]
        adv_amounts = example["adversarial_data"]["amounts"]

        if example["data"]["label"] != example["adversarial_data"]["label"]:
            dataset.append(
                {"transactions": transactions, "amounts": amounts, "label": 0}
            )

            dataset.append(
                {"transactions": adv_transactions, "amounts": adv_amounts, "label": 1}
            )

    return dataset


def main(
    results_path: Path,
    out_data_dir: Path,
    filename: str = "output.json",
    test_size: float = 0.3,
):
    paths = results_path.rglob(filename)

    for path in paths:
        output = load_jsonlines(str(path))
        dataset = create_dataset_from_output(output)

        train, valid = train_test_split(dataset, random_state=23, test_size=test_size)

        attack_name = path.parent.name
        target_name_viasubst_name = path.parent.parent.name
        dataset_name = path.parent.parent.parent.name

        base_dir = (
            out_data_dir
            / dataset_name
            / "adv_detection"
            / attack_name
            / target_name_viasubst_name
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        typer.echo(f"Saving data to {base_dir}")
        write_jsonlines(train, base_dir / "train.jsonl")

        write_jsonlines(valid, base_dir / "valid.jsonl")


if __name__ == "__main__":
    typer.run(main)
