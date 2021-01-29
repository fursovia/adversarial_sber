import typer
import pickle
import numpy as np

from advsber.utils.data import load_jsonlines
from sklearn.preprocessing import KBinsDiscretizer


def main(dataset_name: str):
    
    train = load_jsonlines('../data/' + dataset_name + '/lm/train.jsonl')
    valid = load_jsonlines('../data/' + dataset_name + '/lm/valid.jsonl')
    data = train + valid
    
    amounts = []

    for d in data:
        amounts.extend(d['amounts'])
    
    amounts = np.array(amounts)
    amounts = amounts.reshape(-1, 1)
    
    dis = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='quantile')
    dis.fit(amounts)

    with open('presets/' + dataset_name + '/discretizers/100_quantile', 'wb') as f:
        pickle.dump(dis, f)
    
    dis = KBinsDiscretizer(n_bins=50, encode='ordinal', strategy='quantile')
    dis.fit(amounts)

    with open('presets/' + dataset_name + '/discretizers/50_quantile', 'wb') as f:
        pickle.dump(dis, f)
    
    return

if __name__ == "__main__":
    typer.run(main)