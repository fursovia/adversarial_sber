local clf_target_path = std.extVar("CLF_TARGET_PATH");
local clf_subst_path = std.extVar("CLF_SUBST_PATH");

{
  "data_path": std.extVar("DATA_PATH"),
  "output_path": std.extVar("OUTPUT_PATH"),
  "attacker": {
    "type": "fgsm",
    "classifier_target": {
      "type": "from_archive",
      "archive_file": clf_target_path
    },
    "classifier_subst": {
      "type": "from_archive",
      "archive_file": clf_subst_path
    },
    "reader": {
      "type": "from_archive",
      // we parse reader args from archive
      "archive_file": clf_target_path
    },
    "num_steps": 10,
    "epsilon": 0.3,
    "device": 0
  }
}
}