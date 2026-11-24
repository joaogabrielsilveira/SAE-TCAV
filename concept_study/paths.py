"""Every input path, output path and fixed constant of the concept agreement study."""
from pathlib import Path

import torch

HOME = Path.home()
CODE_ROOT = HOME / "SAE-TCAV"

# Frozen inputs built by earlier runs (read only).
PILOT_ROOT = HOME / "rpi-runs/SAE-TCAV/r1-pilot-fe945-20261009"
PREVIOUS_RUNS = HOME / "rpi-runs/SAE-TCAV/pilot-ideas-followups"
BUILDS_ROOT = PREVIOUS_RUNS / "builds"
POPULATION = CODE_ROOT / "stats/temporal_robustness/_population/prepared.pkl"
MEASUREMENT_STUDY = HOME / "rpi-runs/SAE-TCAV/concept-measurement-reliability"
CAV_FREEZE = MEASUREMENT_STUDY / "20261103-v7c-develop/concept_measurement_reliability/private/freeze.pkl"
DEVELOP_PUBLIC = MEASUREMENT_STUDY / "20261103-v7c-develop/concept_measurement_reliability/public"
EVALUATE_PUBLIC = MEASUREMENT_STUDY / "20261103-v7d-evaluate/concept_measurement_reliability/public"

# Outputs of this study.
OUTPUT = CODE_ROOT / "stats/concept_agreement_study"
PATIENT_TABLE = OUTPUT / "patient_table"
CAV_DIRECTIONS = OUTPUT / "cav_directions.npz"
REPLICATES = OUTPUT / "replicates"
YEAR_INPUT = OUTPUT / "year_input"

# Fixed study choices.
VARIANTS = ("last_3", "all_history")          # last_3 primary, all_history auxiliary
PRIMARY_VARIANT = "last_3"
REFERENCES = tuple(range(2007, 2014))          # same reference years as the November study
LAST_YEAR = 2015                               # last year with embeddings
LAST_OUTCOME_YEAR = 2014                       # 2015 deaths are incomplete: never used as labels
CAV_C = 0.1
ESTIMATORS = ("all_negatives", "legacy")       # all-negative primary, legacy alongside
PRIMARY_ESTIMATOR = "all_negatives"
LOGIT_ESTIMATOR = "all_negatives_logit"        # primary CAV gradient on the log-odds scale: divided by p(1-p)
INFLUENCE_ESTIMATORS = ESTIMATORS + (LOGIT_ESTIMATOR,)
SAE_SEEDS = (42, 135, 215)
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261106
SHUFFLES = 1000
RIDGE_ALPHA = 10.0
RIDGE_ALPHA_CHECKS = (1.0, 100.0)
MIN_TRAINING_ROWS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # one device for every GPU step

# Groups of patients inside a concept-year.
GROUPS = ("A", "R", "R+A+", "R+A-", "R-A+", "neither", "all")
