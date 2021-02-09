local COMMON = import 'common/basic.jsonnet';

local transactions_emb_dim = 64;
local amounts_emb_dim = 32;
local gru_hidden_size = 256;
local gru_num_layers = 1;
local gru_dropout = 0.1;
local bidirectional = true;

{
  "dataset_reader": COMMON["dataset_reader"],
  "train_data_path": std.extVar("CLF_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("CLF_VALID_DATA_PATH"),
  "vocabulary": COMMON["vocabulary"],
  "model": {
    "type": "rtd_encoder",
    "transactions_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": transactions_emb_dim,
          "trainable": true,
          "vocab_namespace": "transactions"
        }
      }
    },
    "amounts_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": amounts_emb_dim,
          "trainable": true,
          "vocab_namespace": "amounts"
        }
      }
    },
    "seq2vec_encoder": {
      "type": "gru",
      "input_size": transactions_emb_dim + amounts_emb_dim,
      "hidden_size": gru_hidden_size,
      "num_layers": gru_num_layers,
      "dropout": gru_dropout,
      "bidirectional": bidirectional
    },
  },
  "data_loader": {
    "shuffle": true,
    "batch_size": 8,
    "num_workers": 0,
    // https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    "pin_memory": true
  },
  "trainer": COMMON["trainer"]
}
