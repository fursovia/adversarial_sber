import json

import typer
import pandas as pd

from advsber.utils.data import load_jsonlines
from advsber.utils.metrics import normalized_accuracy_drop, misclassification_error, probability_drop


def main(output_path: str, save_to: str = typer.Option(None), visualize: bool = typer.Option(False)):
    output = load_jsonlines(output_path)
    output = pd.DataFrame(output).drop(columns="history")

    nad = normalized_accuracy_drop(
        wers=output["wer"],
        y_true=output["attacked_label"],
        y_adv=output["adversarial_label"],
    )
    typer.echo(f"NAD = {nad:.2f}")

    misclf_error = misclassification_error(
        y_true=output["attacked_label"],
        y_adv=output["adversarial_label"],
    )
    typer.echo(f"Misclassification Error = {misclf_error:.2f}")

    prob_drop = probability_drop(true_prob=output["probability"], adv_prob=output["adversarial_probability"])
    typer.echo(f"Probability drop = {prob_drop:.2f}")

    if visualize:
        assert save_to is not None

    if save_to is not None:
        metrics = {"NAD": nad, "ME": misclf_error, "PD": prob_drop}

        with open(save_to, "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    typer.run(main)
