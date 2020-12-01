local COMMON = import 'common/basic.jsonnet';

local transactions_emb_dim = 64;
local amounts_emb_dim = 32;
local num_filters = 64;
local conv_layer_activation = "relu";

{
  "dataset_reader": COMMON["dataset_reader"],
  "train_data_path": std.extVar("CLF_TRAIN_DATA_PATH"),
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
      "type": "cnn",
      "embedding_dim": transactions_emb_dim + amounts_emb_dim,
      "num_filters": num_filters,
      "conv_layer_activation": conv_layer_activation,
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
  "trainer": COMMON["trainer"]
}
