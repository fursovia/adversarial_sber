local COMMON = import 'common/basic.jsonnet';

local transactions_emb_dim = 64;
local lstm_hidden_size = 256;
local lstm_num_layers = 1;
local lstm_dropout = 0.1;
local bidirectional = true;

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
    "seq2vec_encoder": {
      "type": "lstm",
      "input_size": transactions_emb_dim + 1,
      "hidden_size": lstm_hidden_size,
      "num_layers": lstm_num_layers,
      "dropout": lstm_dropout,
      "bidirectional": bidirectional
    },
  },
  "data_loader": COMMON["data_loader"],
  "trainer": COMMON["trainer"]
}

