import json

import typer
import pandas as pd
import numpy as np

from advsber.utils.data import load_jsonlines
from advsber.utils.metrics import (
    normalized_accuracy_drop,
    amount_normalized_accuracy_drop,
    misclassification_error,
    probability_drop,
)


def main(output_path: str, save_to: str = typer.Option(None), visualize: bool = typer.Option(False)):
    output = load_jsonlines(output_path)
    output = pd.DataFrame(output).drop(columns="history")

    y_true = [output["data"][i]["label"] for i in range(len(output))]
    y_adv = [output["adversarial_data"][i]["label"] for i in range(len(output))]
    nad = normalized_accuracy_drop(wers=output["wer"], y_true=y_true, y_adv=y_adv)
    typer.echo(f"NAD = {nad:.2f}")

    misclf_error = misclassification_error(y_true=y_true, y_adv=y_adv)
    typer.echo(f"Misclassification Error = {misclf_error:.2f}")

    prob_drop = probability_drop(true_prob=output["probability"], adv_prob=output["adversarial_probability"])
    typer.echo(f"Probability drop = {prob_drop:.2f}")

    mean_wer = float(np.mean(output["wer"]))
    typer.echo(f"Mean WER = {mean_wer:.2f}")

    added_amounts = []
    for _, row in output.iterrows():
        added_amounts.append(sum(row["adversarial_data"]["amounts"]) - sum(row["data"]["amounts"]))

    anad = amount_normalized_accuracy_drop(added_amounts, y_true=y_true, y_adv=y_adv)
    typer.echo(f"aNAD-1000 = {anad:.2f}")

    if visualize:
        assert save_to is not None

    if save_to is not None:
        metrics = {"NAD": nad, "ME": misclf_error, "PD": prob_drop, "Mean_WER": mean_wer, "aNAD-1000": anad}

        with open(save_to, "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    typer.run(main)
