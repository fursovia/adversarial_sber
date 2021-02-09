import typer
import advsber
from allennlp.data import Vocabulary

def main(dataset_name: str):
    reader = advsber.TransactionsDatasetReader('presets/' + dataset_name + '/discretizers/100_quantile')
    
    train = reader.read('../data/' + dataset_name + '/lm/train.jsonl')
    valid = reader.read('../data/' + dataset_name + '/lm/valid.jsonl')
    
    data = train + valid
    
    tokens_to_add = {"transactions": ["@@MASK@@", "<START>", "<END>"],
                      "amounts": ["<START>", "<END>"]}
    
    vocab = Vocabulary.from_instances(data, tokens_to_add=tokens_to_add)
    vocab.save_to_files('./presets/' + dataset_name + '/vocabs/100_quantile')
    
    return

if __name__ == "__main__":
    typer.run(main)