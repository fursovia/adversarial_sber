import json

import typer
import pandas as pd
import numpy as np
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive

from advsber.utils.data import load_jsonlines
from advsber.utils.metrics import (
    normalized_accuracy_drop,
    amount_normalized_accuracy_drop,
    misclassification_error,
    probability_drop,
    diversity_rate,
    amount_error_rate,
    amount_add_rate,
)


def get_predictor(archive_path: str) -> Predictor:
    archive = load_archive(archive_path, cuda_device=-1)
    predictor = Predictor.from_archive(archive=archive, predictor_name="transactions")
    return predictor


def main(
    output_path: str,
    save_to: str = typer.Option(None),
    target_clf_path: str = typer.Option(None),
):
    output = load_jsonlines(output_path)
    output = pd.DataFrame(output).drop(columns="history")

    if target_clf_path is not None:
        predictor = get_predictor(target_clf_path)
        data = [
            {"transactions": adv_example["transactions"], "amounts": adv_example["amounts"]}
            for adv_example in output["adversarial_data"]
        ]
        preds = predictor.predict_batch_json(data)

        for i, pred in enumerate(preds):
            label = pred["label"]
            prob = pred["probs"][
                predictor._model.vocab.get_token_index(str(output["data"][i]["label"]), namespace="labels")
            ]
            output["adversarial_data"][i]["label"] = int(label)
            output["adversarial_probability"][i] = prob

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

    added_amounts = [1]
    # for _, row in output.iterrows():
    # added_amounts.append(sum(int(row["adversarial_data"]["amounts"])) - sum(int(row["data"]["amounts"])))
    aar = amount_add_rate(output)
    aer = amount_error_rate(output)
    typer.echo(f"am add error = {aar:.2f}")
    typer.echo(f"am error error = {aer:.2f}")
    anad = amount_normalized_accuracy_drop(added_amounts, y_true=y_true, y_adv=y_adv)
    typer.echo(f"aNAD-1000 = {anad:.2f}")
    diversity = diversity_rate(output)
    typer.echo(f"Diversity_rate = {diversity:.2f}")
    if save_to is not None:
        metrics = {
            "NAD": nad,
            "ME": misclf_error,
            "PD": prob_drop,
            "Mean_WER": mean_wer,
            "aNAD-1000": anad,
            "diversity_rate": diversity,
            "aar": aar,
            "aer": aer,
        }
        with open(save_to, "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    typer.run(main)
