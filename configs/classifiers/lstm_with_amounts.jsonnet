
local transactions_emb_dim = 64;
local amounts_emb_dim = 32;
local lstm_hidden_size = 256;
local lstm_num_layers = 1;
local lstm_dropout = 0.1;
local bidirectional = true;

{
  "dataset_reader": {
    "type": "transactions_reader",
    "discretizer_path": std.extVar("DISCRETIZER_PATH"),
    "max_sequence_length": 150,
    "lazy": false
  },
  "train_data_path": std.extVar("CLF_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("CLF_VALID_DATA_PATH"),
  "vocabulary": {
    "type": "from_files",
    "directory": std.extVar("VOCAB_PATH")
  },
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
      "type": "lstm",
      "input_size": transactions_emb_dim + amounts_emb_dim,
      "hidden_size": lstm_hidden_size,
      "num_layers": lstm_num_layers,
      "dropout": lstm_dropout,
      "bidirectional": bidirectional
    },
  },
  "data_loader": {
    "batch_size": 1024,
    "shuffle": true,
    "num_workers": 0,
    // https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    "pin_memory": true
  },
  "trainer": {
    "num_epochs": 50,
    "patience": 2,
    "cuda_device": 0
  }
}
