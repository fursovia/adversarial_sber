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
  "test_data_path": std.extVar("CLF_TEST_DATA_PATH"),
  "validation_data_path": std.extVar("CLF_VALID_DATA_PATH"),
  "vocabulary": COMMON["vocabulary"],
  "model": {
    "type": "transactions_classifier",
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
//  "distributed": {
//    "cuda_devices": [
//      0,
//      1,
//      2,
//    ]
//  },
  "data_loader": COMMON["data_loader"],
  "trainer": COMMON["trainer"],
  "evaluate_on_test": true,
  "random_seed": std.extVar("RANDOM_SEED"),
  "numpy_seed": std.extVar("RANDOM_SEED"),
  "pytorch_seed": std.extVar("RANDOM_SEED")
}