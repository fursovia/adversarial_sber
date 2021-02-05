import typer
import pandas as pd
import numpy as np
import sys
import json
import jsonlines

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from advsber.utils.data import write_jsonlines


def main():
    
    TEST_RATIO = 0.1
    SUBST_RATIO = 0.3
    VALID_RATIO = 0.2
    LM_RATIO = 0.1
    DATASET_PATH = '../data/gender_short'

    MIN_LEN = 3
    MAX_LEN = 50
    
    transactions = pd.read_csv('../data/gender/original/transactions.csv')
    target_data = pd.read_csv('../data/gender/original/gender_train.csv')
    
    target_data_dict = dict(target_data.values)
    data = transactions.rename(columns={'customer_id': 'client_id', 'trans':'small_group', 'amount':'amount_rur'})
    target_data = target_data.rename(columns={'customer_id':'client_id', 'gender':'bins'})
    
    data['week'] = data['tr_datetime'].str.split(' ').apply(lambda x: int(x[0]) // 7)

    clients = data['client_id'].unique()

    my_lovely_data = []

    for client_id in tqdm(clients):
        client_data = data[data['client_id'] == client_id]
        client_weeks = client_data.week.unique()

        for week in client_weeks:
            client_data_week = client_data[client_data['week'] == week]
            my_lovely_data.append(
                {
                    'transactions': client_data_week['mcc_code'].tolist(),
                    'amounts': client_data_week.amount_rur.tolist(),
                    'client_id': client_id, 
                    'week': week
                }
        )
            
    my_lovely_data = pd.DataFrame(my_lovely_data)
    my_lovely_data['label'] = my_lovely_data['client_id'].apply(lambda x: target_data_dict.get(x))
    my_lovely_data = my_lovely_data[~my_lovely_data['label'].isna()]
    my_lovely_data['label'] = my_lovely_data['label'].astype(int)
    my_lovely_data = my_lovely_data[['transactions', 'amounts', 'client_id', 'label']]
    
    lens = my_lovely_data.transactions.apply(lambda x: len(x))
    my_lovely_data = my_lovely_data[(lens >= MIN_LEN) & (lens <= MAX_LEN)]
    
    lm_train, lm_valid = train_test_split(
        my_lovely_data, 
        stratify=my_lovely_data['label'], 
        random_state=124,
        test_size=LM_RATIO
    )

    other_data, test_data = train_test_split(
        my_lovely_data, 
        stratify=my_lovely_data['label'], 
        random_state=123,
        test_size=TEST_RATIO
    )

    target_data, subst_data = train_test_split(
        other_data, 
        stratify=other_data['label'], 
        random_state=123,
        test_size=SUBST_RATIO
    )

    target_data_tr, target_data_val = train_test_split(
        target_data, 
        stratify=target_data['label'], 
        random_state=123,
        test_size=VALID_RATIO
    )

    subst_data_tr, subst_data_val = train_test_split(
        subst_data, 
        stratify=subst_data['label'], 
        random_state=123,
        test_size=VALID_RATIO
    )
    
    write_jsonlines(test_data.to_dict('records'), f'{DATASET_PATH}/test.jsonl')

    write_jsonlines(target_data_tr.to_dict('records'), f'{DATASET_PATH}/target_clf/train.jsonl')
    write_jsonlines(target_data_val.to_dict('records'), f'{DATASET_PATH}/target_clf/valid.jsonl')

    write_jsonlines(subst_data_tr.to_dict('records'), f'{DATASET_PATH}/substitute_clf/train.jsonl')
    write_jsonlines(subst_data_val.to_dict('records'), f'{DATASET_PATH}/substitute_clf/valid.jsonl')

    write_jsonlines(lm_train.to_dict('records'), f'{DATASET_PATH}/lm/train.jsonl')
    write_jsonlines(lm_valid.to_dict('records'), f'{DATASET_PATH}/lm/valid.jsonl')

    return

if __name__ == "__main__":
    typer.run(main)