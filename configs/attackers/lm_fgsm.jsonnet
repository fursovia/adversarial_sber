local clf_path = std.extVar("CLF_PATH");
<<<<<<< HEAD
=======
local lm_path = std.extVar("LM_PATH");
>>>>>>> 01ce875e9c7fb5f190346b2ba61b29e348635816

{
  "data_path": std.extVar("DATA_PATH"),
  "output_path": std.extVar("OUTPUT_PATH"),
  "attacker": {
    "type": "lm_fgsm",
<<<<<<< HEAD
    "lm": {
      "type": "from_archive",
      "archive_file": std.extVar("MASKED_LM_PATH")
    },
=======
>>>>>>> 01ce875e9c7fb5f190346b2ba61b29e348635816
    "classifier": {
      "type": "from_archive",
      "archive_file": clf_path
    },
<<<<<<< HEAD
=======
    "lm": {
      "type": "from_archive",
      "archive_file": lm_path
    },
>>>>>>> 01ce875e9c7fb5f190346b2ba61b29e348635816
    "reader": {
      "type": "from_archive",
      // we parse reader args from archive
      "archive_file": clf_path
    },
<<<<<<< HEAD
    "lm_threshold": 3.0,
    "num_steps": 20.0,
    "epsilon": 1.0,
    "device": 0
  }
}
=======
    "lm_threshold": 1.5,
    "num_steps": 30,
    "epsilon": 1.0,
    "device": 0
  }
}
>>>>>>> 01ce875e9c7fb5f190346b2ba61b29e348635816
