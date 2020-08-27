{
  "data_path": std.extVar("DATA_PATH"),
  "output_path": std.extVar("OUTPUT_PATH"),
  "attacker": {
    "type": "sampling_fool",
    "masked_lm_dir": std.extVar("MASKED_LM_PATH"),
    "classifier_dir": std.extVar("CLF_PATH"),
    "num_samples": 100,
    "temperature": 1.0,
    "device": -1,
  }
}