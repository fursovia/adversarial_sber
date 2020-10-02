import json
import typer
import jsonlines

from allennlp.common.file_utils import cached_path

def main(train_path: str, path_save: str, num_samples: int=typer.Option(None), adv_data_path: str=typer.Option(None)):
    if not adv_data_path is None:
        with jsonlines.open(path_save, "w") as writer:
            with open(cached_path(adv_data_path), "r") as reader:
                for item in reader.readlines():
                    item = json.loads(item)
                    label = item['data']['label']
                    data_adv = item["adversarial_data"]
                    data_adv['label'] = label
                    writer.write(data_adv)

    with jsonlines.open(path_save, "w") as writer:
        with open(cached_path(train_path), "r") as reader:
            if not num_samples is None:
                count_ = 0

            for item in reader.readlines():
                writer.write(item)
                if not num_samples is None:
                    if count_ > num_samples: break
                    count_ += 1

    return

if __name__ == "__main__":
    typer.run(main)