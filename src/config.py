"""
Configuration module for the ECG Arrhythmia Detection and Classification project.
Contains all project paths, signal parameters, annotation mappings, and model hyperparameters.
"""

import os
from pathlib import Path

# ==========================================
# 1. Directory and Path Settings
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "mitdb"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Ensure all target output directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Complete 48 MIT-BIH Arrhythmia Records
# ==========================================
# 100-series: Routine clinical recordings
SERIES_100 = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124"
]

# 200-series: Complex ventricular, junctional, and supraventricular arrhythmias
SERIES_200 = [
    "200", "201", "202", "203", "205", "207", "208", "209", "210", "212",
    "213", "214", "215", "217", "219", "220", "221", "222", "223", "228",
    "230", "231", "232", "233", "234"
]

ALL_48_RECORDS = sorted(SERIES_100 + SERIES_200)

# Pacemaker records (contain artificial pacing artifacts)
PACEMAKER_RECORDS = ["102", "104", "107", "217"]

# ==========================================
# 3. Sampling and Preprocessing Parameters
# ==========================================
SAMPLING_RATE = 360  # Hz (MIT-BIH standard sampling rate)

# Butterworth Bandpass Filter (removes baseline wander & high-frequency muscle noise)
BANDPASS_LOWCUT = 0.5   # Hz (Preserves ST segments while removing breathing wander)
BANDPASS_HIGHCUT = 45.0 # Hz (Retains QRS frequency band while removing powerline/EMG)
FILTER_ORDER = 3        # Applied bidirectionally (filtfilt) -> effective 6th order, zero-phase

# Notch filter for powerline hum removal (if present)
NOTCH_FREQ = 60.0       # Hz (US powerline standard)
NOTCH_Q = 30.0          # Quality factor

# ==========================================
# 4. Heartbeat Windowing & Segmentation
# ==========================================
# Heartbeat window around R-peak (at 360 Hz)
# 90 samples pre-R  = ~250 ms (captures P-wave and PR segment)
# 130 samples post-R = ~361 ms (captures S-T segment and T-wave)
PRE_R_SAMPLES = 90
POST_R_SAMPLES = 130
WINDOW_SIZE = PRE_R_SAMPLES + POST_R_SAMPLES  # 220 samples (~611 ms)

# R-peak tolerance window for Pan-Tompkins evaluation (+/- 50 ms)
R_PEAK_TOLERANCE_MS = 50.0
R_PEAK_TOLERANCE_SAMPLES = int((R_PEAK_TOLERANCE_MS / 1000.0) * SAMPLING_RATE)  # 18 samples

# ==========================================
# 5. AAMI Standard Beat Classification Mapping
# ==========================================
# Mapping 15+ MIT-BIH annotation codes into 5 AAMI standard classes:
# N: Normal, S: Supraventricular ectopic, V: Ventricular ectopic, F: Fusion, Q: Unknown/Paced
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

MIT_TO_AAMI = {
    # N - Normal and Bundle Branch Block beats
    "N": "N",  # Normal beat
    "L": "N",  # Left bundle branch block beat
    "R": "N",  # Right bundle branch block beat
    "e": "N",  # Atrial escape beat
    "j": "N",  # Nodal (junctional) escape beat
    
    # S - Supraventricular Ectopic Beats (SVEB)
    "A": "S",  # Atrial premature beat
    "a": "S",  # Aberrated atrial premature beat
    "J": "S",  # Nodal (junctional) premature beat
    "S": "S",  # Supraventricular premature beat
    
    # V - Ventricular Ectopic Beats (VEB)
    "V": "V",  # Premature ventricular contraction
    "E": "V",  # Ventricular escape beat
    "!": "V",  # Ventricular flutter wave
    
    # F - Fusion beats
    "F": "F",  # Fusion of ventricular and normal beat
    
    # Q - Paced and unclassifiable beats
    "/": "Q",  # Paced beat
    "f": "Q",  # Fusion of paced and normal beat
    "Q": "Q",  # Unclassifiable beat
}

# Non-beat markers to exclude from classification
NON_BEAT_SYMBOLS = {
    "[", "]", "x", "(", ")", "p", "t", "u", "`", "'", "^", "|", "~", "+", "s", "T", "*", "D", "="
}

# Class integer encoding for model training
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(AAMI_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for cls_name, idx in CLASS_TO_IDX.items()}

# ==========================================
# 6. Strict Patient-Wise Dataset Partitioning
# ==========================================
# Based on the inter-patient paradigm (de Chazal et al., IEEE TBME 2004)
# Note: Record 201 and 202 are from the SAME subject. To prevent patient leakage,
# both 201 and 202 are grouped together into the training/validation pool.
# All 48 records are partitioned cleanly with zero patient overlap.

PATIENT_SPLIT = {
    # DS1 - Training Set (24 records, ~50% of database)
    "train": [
        "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
        "122", "124", "201", "202", "203", "205", "207", "208", "209", "215",
        "220", "223", "230", "102"
    ],
    # Validation Set (4 records for hyperparameter tuning)
    "val": [
        "104", "107", "217", "222"
    ],
    # DS2 - Independent Test Set (20 records, ~42% of database)
    "test": [
        "100", "103", "105", "111", "113", "117", "121", "123", "200", "210",
        "212", "213", "214", "219", "221", "228", "231", "232", "233", "234"
    ]
}

# Verify partition covers all 48 records with no duplicates
_all_split = PATIENT_SPLIT["train"] + PATIENT_SPLIT["val"] + PATIENT_SPLIT["test"]
assert sorted(_all_split) == ALL_48_RECORDS, "Patient split must contain all 48 unique recordings!"

# ==========================================
# 7. Model Hyperparameters
# ==========================================
# Feature framing for HMM: Each 220-sample beat is divided into sequential sub-windows
HMM_N_FRAMES = 11        # 11 frames of 20 samples each (captures temporal progression)
HMM_N_STATES = 5         # Corresponds to: P-wave, PR-segment, QRS-complex, ST-segment, T-wave
HMM_COVAR_TYPE = "diag"  # Diagonal covariance for stability and computational efficiency
HMM_N_ITER = 50          # Baum-Welch (EM) iterations
HMM_MIN_COVAR = 1e-3     # Regularization floor to prevent singular covariance collapse

RANDOM_SEED = 42
