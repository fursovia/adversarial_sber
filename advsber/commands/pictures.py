
from collections import Counter
from typing import List, Any, Dict
import jsonlines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import typer

def load_jsonlines(path: str) -> List[Dict[str, Any]]:
    data = []
    with jsonlines.open(path, "r") as reader:
        for items in reader:
            data.append(items)
    return data


def adversarial_tokens_amounts(t_true: List[int], t_adv: List[str], a_true=None, a_adv=None) -> List[int]:
    t_ins = []
    advers_amount_distr = []
    special_tokens = ['@@PADDING@@', '@@MASK@@', "@@UNKNOWN@@", "<START>", "<END>"]
    for i in range(len(t_adv)):
        for t in range(len(t_adv[i])):
            if t_adv[i][t] not in special_tokens:
                if (t > len(t_true[i]) - 1):
                    t_ins.append(int(t_adv[i][t]))
                    if a_true:
                        advers_amount_distr.append(abs(a_adv[i][t]))
                else:
                    if (int(t_adv[i][t]) != int(t_true[i][t])):
                        t_ins.append(int(t_adv[i][t]))
                        if a_true:
                            advers_amount_distr.append(abs(a_adv[i][t]))
            else:
                continue
        
    return t_ins, advers_amount_distr


def plot_statistics(path_to_adversarial_data, path_to_save_folder):
    
    output = load_jsonlines(path_to_adversarial_data)
    output = pd.DataFrame(output).drop(columns="history")

    t_true = [output["data"][i]["transactions"] for i in range(len(output))]
    t_adv = [output["adversarial_data"][i]["transactions"] for i in range(len(output))]
    t_ins = adversarial_tokens_amounts(t_true, t_adv)[0] 
    
    t_true_full = [t for sublist in t_true for t in sublist]
    tokens_freq = dict(Counter(t_true_full))
    tokens_freq_sorted =  {k: v for k, v in sorted(tokens_freq.items(), key=lambda item: item[1])}
    t_ins_frequency = dict(Counter(t_ins))
    
    ins_common_freq = []
    ins_rest_freq = []
    for t in tokens_freq_sorted.keys():
        if t in t_ins_frequency.keys():
            ins_common_freq.append(t_ins_frequency[t])
        else:
            ins_common_freq.append(0)
            
    for t in t_ins_frequency.keys():
        if t not in tokens_freq_sorted.keys():
            ins_rest_freq.append(t_ins_frequency[t])

    length_common_zone = len(tokens_freq)
    length_rest_zone = len(ins_rest_freq) 
    ins_rest_freq.sort()
    
    fig, ax =  plt.subplots(1, 1)
    fig.set_figheight(10)
    fig.set_figwidth(13)
    
    ax.bar(np.arange(length_common_zone+length_rest_zone), ins_common_freq+ins_rest_freq, color='orange', label='Distribution of inserted adversarial tokens', alpha = 1, width=1)
    ax.bar(np.arange(length_common_zone), tokens_freq_sorted.values(), color='green', label='Distribution of initial tokens', alpha = 0.5, width=1)
    ax.set_yscale("log")
    ax.tick_params(labelsize=25)
    ax.set_xlabel('Value of token', fontsize=20)
    ax.set_ylabel('Frequency', fontsize=25)
    ax.set_xticklabels([])
    ax.legend(fontsize=25)
    ax.grid()    
    plt.savefig(path_to_save_folder + "tokens_distribution.pdf", bbox_inches='tight')
    plt.close(fig)  


def plot_amounts(path_to_adversarial_data, path_to_save_folder):
    
    output = load_jsonlines(path_to_adversarial_data)
    output = pd.DataFrame(output).drop(columns="history")

    t_true = [output["data"][i]["transactions"] for i in range(len(output))]
    t_adv = [output["adversarial_data"][i]["transactions"] for i in range(len(output))]
    a_true = [output["data"][i]["amounts"] for i in range(len(output))]
    a_adv = [output["adversarial_data"][i]["amounts"] for i in range(len(output))]
    
    initial_amount_distr = [abs(a) for sublist in a_true for a in sublist]
    advers_amount_distr = adversarial_tokens_amounts(t_true, t_adv, a_true, a_adv)[1]
    
    fig, ax =  plt.subplots(1, 1)
    fig.set_figheight(10)
    fig.set_figwidth(13)
       
    ax.hist(advers_amount_distr, color='orange', alpha=1, label="Distribution of adversarial amounts")
    ax.hist(initial_amount_distr, color='green', alpha=0.45, label="Distribution of initial amounts")
    ax.legend(fontsize=25)
    ax.set_yscale("log")
    ax.set_xscale("log") 
    ax.tick_params(labelsize=25)
    ax.set_xlabel("Value of amount", fontsize=20)
    ax.set_ylabel("Frequency", fontsize=25)
    ax.grid()
    plt.savefig(path_to_save_folder + "amounts_distribution.pdf", bbox_inches='tight')
    plt.close(fig)                         

def main(dataset_name: str, attack_name: str, subst_name: str, targ_name: str):
    path_to_adversarial_data = "../experiments/attacks/" + dataset_name + "/targ_" + targ_name + "/subst_" + subst_name + "/" + attack_name + "/adversarial.json"
    path_to_save_folder = "../experiments/attacks/" + dataset_name + "/targ_" + targ_name + "/subst_" + subst_name + "/" + attack_name + "/"
    plot_statistics(path_to_adversarial_data, path_to_save_folder)
    plot_amounts(path_to_adversarial_data, path_to_save_folder)
    
if __name__ == "__main__":
    typer.run(main)
