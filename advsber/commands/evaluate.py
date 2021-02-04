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
    calculate_perplexity,
)


def get_predictor(archive_path: str) -> Predictor:
    archive = load_archive(archive_path, cuda_device=-1)
    predictor = Predictor.from_archive(archive=archive, predictor_name="transactions")
    return predictor


def main(
        output_path: str,
        save_to: str = typer.Option(None),
        target_clf_path: str = typer.Option(None),
        lm_path: str = typer.Option(None)
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

    added_amounts = []
    for _, row in output.iterrows():
        added_amounts.append(sum(row["adversarial_data"]["amounts"]) - sum(row["data"]["amounts"]))

    anad = amount_normalized_accuracy_drop(added_amounts, y_true=y_true, y_adv=y_adv)
    typer.echo(f"aNAD-1000 = {anad:.2f}")
    try:
        diversity = diversity_rate(output)
        diversity = round(diversity, 3)
    except ValueError:
        diversity = None
    typer.echo(f"Diversity_rate = {diversity}")

    if lm_path is not None:
        perplexity = calculate_perplexity(
            [adv_example["transactions"] for adv_example in output["adversarial_data"]],
            get_predictor(lm_path)
        )
        typer.echo(f"perplexity = {perplexity}")
    else:
        perplexity = None

    if save_to is not None:
        metrics = {
            "NAD": round(nad, 3),
            "ME": round(misclf_error, 3),
            "PD": round(prob_drop, 3),
            "Mean_WER": round(mean_wer, 3),
            "aNAD-1000": round(anad, 3),
            "diversity_rate": diversity,
            "perplexity": perplexity
        }
        with open(save_to, "w") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    typer.run(main)
