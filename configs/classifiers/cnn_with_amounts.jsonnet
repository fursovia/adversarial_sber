{
  "aataset_reader": {
    "type": "transactions_reader",
    "discretizer_path": std.extVar("DISCRETIZER_PATH"),
    "max_sequence_length": 150,
    "lazy": false
  },
  "train_data_path": std.extVar("LM_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("LM_VALID_DATA_PATH"),
  "vocabulary": {
//    "max_vocab_size": {
//      "tokens": 50000
//    },
    "tokens_to_add": {
      "transactions": [
        "@@MASK@@",
        "<START>",
        "<END>"
      ],
    "amounts": [
        "<START>",
        "<END>"
      ]
    }
   },
  "model": {
    "type": "transactions_classifier",
    "transactions_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 100,

          "trainable": true,
          "vocab_namespace": "transactions"
        }
      }
    },
    "amounts_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 100,
          "trainable": true,
          "vocab_namespace": "amounts"
        }
      }
    },
    "seq2vec_encoder": {
        "type": "cnn",
        "embedding_dim": 200,
        "num_filters": 10,
        "ngram_filter_sizes": [
          3,
          3,
          3
        ],
        "conv_layer_activation" : "relu"
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
      "patience": 3,
      "cuda_device": 0
    }
}
