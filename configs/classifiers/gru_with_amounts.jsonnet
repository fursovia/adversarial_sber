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
          "embedding_dim": 64,
          "trainable": true,
          "vocab_namespace": "transactions"
        }
      }
    },
    "amounts_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 32,
          "trainable": true,
          "vocab_namespace": "amounts"
        }
      }
    },
    "seq2seq_encoder": {
      "type": "gru",
      "input_size": 96,
      "hidden_size": 128,
      "num_layers": 1,
      "dropout": 0.1,
      "bidirectional": true
    },
    "seq2vec_encoder": {
      "type": "bag_of_embeddings",
      "embedding_dim": 256,
      "averaged": true
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
