{
  "dataset_reader": {
    "type": "transactions_reader",
    "discretizer_path": std.extVar("DISCRETIZER_PATH"),
    "max_sequence_length": 150,
    "lazy": false
  },
  "vocabulary": {
    "type": "from_files",
    "directory": std.extVar("VOCAB_PATH")
  },
  "data_loader": {
    "shuffle": true,
    "batch_size": 1024,
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
