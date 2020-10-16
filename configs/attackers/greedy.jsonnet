local clf_path = std.extVar("CLF_PATH");

{
  "data_path": std.extVar("DATA_PATH"),
  "output_path": std.extVar("OUTPUT_PATH"),
  "attacker": {
    "type": "greedy",
    "classifier": {
      "type": "from_archive",
      "archive_file": clf_path
    },
    "reader": {
      "type": "from_archive",
      // we parse reader args from archive
      "archive_file": clf_path
    },
    "num_steps": null,
    "device": 0
  }
}