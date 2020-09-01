{
  "data_path": std.extVar("DATA_PATH"),
  "output_path": std.extVar("OUTPUT_PATH"),
  "attacker": {
    "type": "concat_sampling_fool",
    "masked_lm_archive_path": std.extVar("MASKED_LM_PATH"),
    "classifier_archive_path": std.extVar("CLF_PATH"),
    "position": "end",
    "num_tokens_to_add": 2,
    "num_samples": 100,
    "temperature": 1.0,
    "device": -1,
  }
}