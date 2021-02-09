import pandas as pd
import numpy as np
import json
import jsonlines
import typer

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_slice_subsample(sub_data, cnt_min, cnt_max, split_count):
    sub_datas = []
    cnt_min = cnt_min if len(sub_data) > cnt_max else int(cnt_min*len(sub_data)/cnt_max)
    cnt_max = cnt_max if len(sub_data) > cnt_max else len(sub_data)-1
    split_count = split_count if len(sub_data) > cnt_max else int(len(sub_data)/cnt_max*split_count)
    for i in range(0, split_count):
        if cnt_min < cnt_max: 
            T_i = np.random.randint(cnt_min, cnt_max)
            s = np.random.randint(0, len(sub_data)-T_i-1)
            S_i = sub_data[s:s+T_i-1]
            sub_datas.append(S_i)
    return sub_datas



def create_set(name, data, target):
    len_ = len(np.unique(target.client_id))
    dict_data = {}
    with jsonlines.open(name, "w") as writer:
        for client_id in tqdm(np.unique(target.client_id)):
            sub_data = data[data['client_id']==client_id]
            sub_data_target = target[target['client_id']==client_id]
            sub_datas = split_slice_subsample(sub_data, 25, 150, 30)
            for loc_data in sub_datas:
                if len(loc_data.small_group):
                    loc_dict = {"transactions": list(loc_data.small_group),
                                "amounts": list(loc_data.amount_rur),
                                "label": int(sub_data_target.bins),
                                "client_id": int(client_id)}
                    writer.write(loc_dict) 
                
    return


def split_data(data, target_data, dir_):
    target_data_train, target_data_valid = train_test_split(target_data, test_size=0.2, random_state=10, shuffle=True)
    print('Create train set...')
    create_set(dir_+'/'+'train.jsonl', data, target_data_train)
    print('Create valid set...')
    create_set(str(dir_)+'/'+'valid.jsonl', data, target_data_valid)
    return


def main():
    
    transactions = pd.read_csv('../data/gender/original/transactions.csv')
    target_data = pd.read_csv('../data/gender/original/gender_train.csv')
    
    transactions.term_id = transactions.term_id.fillna("UNK")
    transactions['trans'] = "mcc" + transactions['mcc_code'].astype(str)
    
    data = transactions.rename(columns={'customer_id': 'client_id', 'trans':'small_group', 'amount':'amount_rur'})
    target_data = target_data.rename(columns={'customer_id':'client_id', 'gender':'bins'})
    
    #change transaction to numbers
    keys = np.unique(data.small_group)
    new_values = np.arange(0,len(keys), dtype=int)
    dictionary = dict(zip(keys, new_values))
    new_column = [dictionary[key] for key in list(data.small_group)]
    data.small_group = new_column
    
    target_data_test_sub, target_data_targetclf = train_test_split(target_data, test_size=0.65, random_state=10, shuffle=True)
    target_data_subclf, target_data_test = train_test_split(target_data_test_sub, test_size=2./7, random_state=10, shuffle=True)
    
    print('Create test set...')    
    create_set('../data/gender/test.jsonl', data, target_data_test)
    print('')
    split_data(data, target_data, '../data/gender/lm')
    split_data(data, target_data_subclf, '../data/gender/substitute_clf')
    split_data(data, target_data_targetclf, '../data/gender/target_clf')
    
    
    return

if __name__ == "__main__":
    typer.run(main)