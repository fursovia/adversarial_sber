local TOKEN_INDEXER = {
    "tokens": {
        "type": "single_id",
        "start_tokens": [
          "<START>"
        ],
        "end_tokens": [
          "<END>"
        ],
        // should be set to the maximum value of `ngram_filter_sizes`
        "token_min_padding_length": 5
      }
};

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
    },
  "model": {
    "type": "BasicClassifier",
    "transactions_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 128,
          "trainable": true,
          "vocab_namespace": "transactions"
        }
      }
    },
    "amounts_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 128,
          "trainable": true,
          "vocab_namespace": "amounts"
        }
      }
    },
    "seq2seq_encoder": {
        "type": "gru",
        "input_size": 256,
        "hidden_size": 128,
        "num_layers": 1,
        "dropout": 0.1,
        "bidirectional": true
    }
    "tokens_masker": {
      "type": "tokens_masker",
      "mask_probability": 0.3,
      "replace_probability": 0.1
    }
  },
  "data_loader": {
    "batch_size": 64
  },
//  "distributed": {
//    "master_port": 29555,
//    "cuda_devices": [
//      0,
//      1
//    ]
//  },
  "trainer": {
    "num_epochs": 50,
    "patience": 2,
    "cuda_device": 0
  }
}