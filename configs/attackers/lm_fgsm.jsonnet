local clf_path = std.extVar("CLF_PATH");

{
  "data_path": std.extVar("DATA_PATH"),
  "output_path": std.extVar("OUTPUT_PATH"),
  "attacker": {
    "type": "lm_fgsm",
    "lm": {
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
    "lm_threshold": 3.0,
    "num_steps": 20.0,
    "epsilon": 1.0,
    "device": 0
  }
}
