local clf_path = std.extVar("CLF_PATH");
local lm_path = std.extVar("LM_PATH");

{
  "data_path": std.extVar("DATA_PATH"),
  "output_path": std.extVar("OUTPUT_PATH"),
  "attacker": {
    "type": "lm_fgsm",
    "classifier": {
      "type": "from_archive",
      "archive_file": clf_path
    },
    "lm": {
      "type": "from_archive",
      "archive_file": lm_path
    },
    "reader": {
      "type": "from_archive",
      // we parse reader args from archive
      "archive_file": clf_path
    },
    "lm_threshold": 1.5,
    "num_steps": 30,
    "epsilon": 1.0,
    "device": 0
  }
}