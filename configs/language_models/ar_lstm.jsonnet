{
  "dataset_reader": {
    "type": "transactions_reader",
    "discretizer_path": std.extVar("DISCRETIZER_PATH"),
    "max_sequence_length": 150,
    "lazy": false
  },
  "train_data_path": std.extVar("LM_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("LM_VALID_DATA_PATH"),
  "vocabulary": {
    "type": "from_files",
    "directory": std.extVar("VOCAB_PATH")
  },
  "model": {
    "type": "autoregressive_language_model",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 64,
          "trainable": true,
          "vocab_namespace": "transactions"
        }
      }
    },
    "contextualizer": {
      "type": "lstm",
      "input_size": 64,
      "hidden_size": 128,
      "num_layers": 1,
      "bidirectional": false
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
    "num_epochs": 500,
    "patience": 2,
    "cuda_device": -1
  }
}
