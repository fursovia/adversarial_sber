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
            dataset.append({"transactions": transactions, "amounts": amounts, "label": 0})

            dataset.append({"transactions": adv_transactions, "amounts": adv_amounts, "label": 1})

    return dataset


def main(
        results_path: str, test_size: float = 0.5,
):
    
    train_output = load_jsonlines(results_path +'/train_adv_detection.json')
    valid_output = load_jsonlines(results_path + '/valid_adv_detection.json')
    
    train_dataset = create_dataset_from_output(train_output)
    valid_dataset = create_dataset_from_output(valid_output)
    valid, test = train_test_split(valid_dataset, random_state=23, test_size=test_size)

    write_jsonlines(train_dataset, results_path + "/train_adv_detection_dataset.jsonl")
    write_jsonlines(valid, results_path + "/valid_adv_detection_dataset.jsonl")
    write_jsonlines(test, results_path + "/test_adv_detection_dataset.jsonl")


if __name__ == "__main__":
    typer.run(main)
