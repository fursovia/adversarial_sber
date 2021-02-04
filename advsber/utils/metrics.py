import functools
from typing import List, Any, Dict
import math

from allennlp.predictors import Predictor
from Levenshtein import distance as lvs_distance


@functools.lru_cache(maxsize=5000)
def word_error_rate(sequence_a: str, sequence_b: str) -> int:
    # taken from https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py
    b = set(sequence_a.split() + sequence_b.split())
    word2char = dict(zip(b, range(len(b))))

    w1 = [chr(word2char[w]) for w in sequence_a.split()]
    w2 = [chr(word2char[w]) for w in sequence_b.split()]

    return lvs_distance("".join(w1), "".join(w2))


def word_error_rate_on_sequences(sequence_a: List[int], sequence_b: List[int]) -> int:
    sequence_a = list(map(str, sequence_a))
    sequence_b = list(map(str, sequence_b))

    sequence_a = " ".join(sequence_a)
    sequence_b = " ".join(sequence_b)
    return word_error_rate(sequence_a, sequence_b)


def normalized_accuracy_drop(wers: List[int], y_true: List[int], y_adv: List[int], gamma: float = 1.0,) -> float:
    assert len(y_true) == len(y_adv)
    nads = []
    for wer, lab, alab in zip(wers, y_true, y_adv):
        if wer > 0 and lab != alab:
            nads.append(1 / wer ** gamma)
        else:
            nads.append(0.0)

    return sum(nads) / len(nads)


def misclassification_error(y_true: List[int], y_adv: List[int],) -> float:
    misses = []
    for lab, alab in zip(y_true, y_adv):
        misses.append(float(lab != alab))

    return sum(misses) / len(misses)


def probability_drop(true_prob: List[float], adv_prob: List[float],) -> float:
    prob_diffs = []
    for tp, ap in zip(true_prob, adv_prob):
        prob_diffs.append(tp - ap)

    return sum(prob_diffs) / len(prob_diffs)


def amount_normalized_accuracy_drop(
    added_amounts: List[float], y_true: List[int], y_adv: List[int], target_amount: float = 1000.0,
) -> float:
    assert len(y_true) == len(y_adv)
    nads = []
    for amount, lab, alab in zip(added_amounts, y_true, y_adv):
        penalty = amount / target_amount
        penalty = penalty if penalty > 1.0 else 1.0
        if lab != alab:
            nads.append(1 / penalty)
        else:
            nads.append(0.0)

    return sum(nads) / len(nads)


def diversity_rate(output: List[Dict[str, Any]]) -> float:
    y_true = [output["data"][i]["transactions"] for i in range(len(output))]
    y_adv = [output["adversarial_data"][i]["transactions"] for i in range(len(output))]
    y_ins = []
    for i in range(len(y_adv)):
        for t in range(len(y_adv[i])):
            # if addition
            if t > len(y_true[i]) - 1:
                if y_adv[i][t] == "<START>":
                    y_adv[i][t] = 0
                if y_adv[i][t] == "<END>":
                    y_adv[i][t] = -1
                y_ins.append(int(y_adv[i][t]))
            # if insertion
            else:
                if int(y_adv[i][t]) != int(y_true[i][t]):
                    y_ins.append(int(y_adv[i][t]))
    return len(list(dict.fromkeys(y_ins))) / len(y_ins)


def calculate_perplexity(transactions: List[List[int]], predictor: Predictor) -> float:
    perplexities = []
    for tr in transactions:
        out = predictor.predict_json(
            {
                "transactions": tr,
                "amounts": tr
            }
        )

        perp = math.exp(out["loss"])
        perplexities.append(perp)

    return sum(perplexities) / len(perplexities)
