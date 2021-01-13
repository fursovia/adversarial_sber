import json
import typer
import jsonlines

from allennlp.common.file_utils import cached_path

def main(train_path: str, path_save: str, advpath: str, numsample: int):
    if not advpath is None:
        with jsonlines.open(path_save, "w") as writer:
            
            with open(cached_path(train_path), "r") as reader:
                for item in reader.readlines():
                    item = json.loads(item)
                    writer.write(item)
            
            count_ = 0
            with open(cached_path(advpath), "r") as reader:
                for item in reader.readlines():
                    item = json.loads(item)
                    label = item['data']['label']
                    data_adv = item["adversarial_data"]
                    data_adv['label'] = label
                    writer.write(data_adv)
                    if not numsample is None:
                        if count_ > numsample: break
                        count_ += 1               
    return

if __name__ == "__main__":
    typer.run(main)