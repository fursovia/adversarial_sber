local clf_path = std.extVar("CLF_PATH");

{
  "data_path": std.extVar("DATA_PATH"),
  "output_path": std.extVar("OUTPUT_PATH"),
  "attacker": {
    "type": "concat_fgsm",
    "classifier": {
      "type": "from_archive",
      "archive_file": clf_path
    },
    "reader": {
      "type": "from_archive",
      // we parse reader args from archive
      "archive_file": clf_path
    },
    "num_steps": 10,
    "epsilon": 0.01,
    "position": "end",
    "num_tokens_to_add": 2,
    "total_amount": 5000.0,
    "device": 0
  }
}