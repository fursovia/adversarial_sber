local clf_path = std.extVar("CLF_PATH");

{
  "data_path": std.extVar("DATA_PATH"),
  "output_path": std.extVar("OUTPUT_PATH"),
  "attacker": {
    "type": "sampling_fool",
    "masked_lm": {
      "type": "from_archive",
      "archive_file": std.extVar("MASKED_LM_PATH")
    },
    "classifier": {
      "type": "from_archive",
      "archive_file": clf_path
    },
    "reader": {
      "type": "from_archive",
      // we parse reader args from archive
      "archive_file": clf_path
    },
    "num_samples": 100,
    "temperature": 1.5,
    "device": -1
  }
}