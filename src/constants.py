import os

DATASET_DIR = "datasets"
INITIAL_CONDITIONS_PATH = os.path.join(DATASET_DIR, "initial_conditions.csv")

#27 batches from a 3 level full factorial design + 3 replicates
BATCH_IDS = [1, 10, 19, 64, 73, 82, 127, 136, 145,442, 451, 460, 505, 514, 515, 516, 523,568, 577, 586, 883, 892, 901,
             946, 955,964, 1009, 1018, 1027, 1028]

# Economics
DOWNTIME_D = 1
GLUCOSE_COST = 10        # $/L per mM
GLUTAMINE_COST = 20      # $/L per mM
VCD_COST = 0.0005        # $/L per cell/mL
MAB_VALUE = 1000         # $/L per mg/L

# Model
FEATURES = ["Initial Glucose [mM]", "Initial Glutamine [mM]", "Initial VCD [cells/mL]", "Time [d]"]
TARGET = "mAb [mg/L]"
CV_FOLDS = 5
POLY_DEGREES = [1, 2, 3, 4, 5, 6]
KNN_NEIGHBORS = list(range(1, 41))

# Selected model
SELECTED_DEGREE = 3

# Search grid for the optimization
GLUCOSE_LEVELS = [5, 10, 15, 20, 25, 30, 35]
GLUTAMINE_LEVELS = [1, 2, 3, 4, 5, 6, 7]
VCD_LEVELS = [200000, 400000, 600000, 800000, 1000000, 1200000, 1400000]
HARVEST_TIMES_D = [i / 3 for i in range(1, 85)]   # every 8 h, up to 28 d
Z_95 = 1.96