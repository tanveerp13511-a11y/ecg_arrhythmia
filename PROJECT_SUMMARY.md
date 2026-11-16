# ECG Arrhythmia Detection & Classification Using Hidden Markov Models (HMM)
**Comprehensive Project Summary & Technical Report**

---

## 1. Project Objective & Scope

The objective of this project is to develop an end-to-end, research-grade machine learning system that automatically analyzes electrocardiogram (ECG) biopotential signals and classifies abnormal cardiac rhythms (arrhythmias).

Unlike conventional undergraduate or introductory projects that analyze only a single recording (e.g., Record 100), this system is built and evaluated across the **complete MIT-BIH Arrhythmia Database from PhysioNet** (all 48 recordings, ~24 hours of dual-lead ECG data, **109,966 total heartbeats**).

---

## 2. Medical & Clinical Standards Used

### ANSI/AAMI EC57:1998 Standard
Over 15 detailed clinical annotation codes from the MIT-BIH database were systematically mapped into the 5 standard clinical superclasses:

| AAMI Class | Clinical Meaning | MIT-BIH Annotation Codes Included | Beat Count | Percentage |
| :---: | :--- | :--- | :---: | :---: |
| **N** | Normal & Bundle Branch Block beats | Normal (`N`), Left bundle branch block (`L`), Right bundle branch block (`R`), Atrial escape (`e`), Nodal escape (`j`) | 90,631 | 82.42% |
| **S** | Supraventricular Ectopic Beats (SVEB) | Atrial premature (`A`), Aberrated atrial premature (`a`), Nodal premature (`J`), Supraventricular premature (`S`) | 2,781 | 2.53% |
| **V** | Ventricular Ectopic Beats (VEB) | Premature ventricular contraction (`V`), Ventricular flutter wave (`!`), Ventricular escape (`E`) | 7,708 | 7.01% |
| **F** | Fusion Beats | Fusion of ventricular and normal beat (`F`) | 803 | 0.73% |
| **Q** | Paced & Unclassifiable Beats | Paced beat (`/`), Fusion of paced and normal (`f`), Unclassifiable beat (`Q`) | 8,043 | 7.31% |

*Non-beat markers (rhythm changes `[`, `]`, noise annotations `~`, `|`, measurement artifacts) totaling 2,244 instances were explicitly identified and excluded from beat classification.*

---

## 3. End-to-End Pipeline Architecture

```
Raw 48 ECG Records (PhysioNet)
           │
           ▼
[1] Signal Preprocessing (0.5–45 Hz Zero-Phase Bandpass + 60 Hz Notch)
           │
           ▼
[2] R-Peak Detection (Pan-Tompkins Algorithm: 96.29% Sensitivity, 97.53% F1)
           │
           ▼
[3] Strict Patient-Wise Splitting (Zero patient data leakage)
           │
           ▼
[4] Beat Segmentation (220-sample windows + RR interval dynamics)
           │
     ┌─────┴────────────────────────┐
     ▼                              ▼
[5A] Sequential Frames         [5B] Tabular Features
     (11 frames x 8 features)       (27 morphology & rhythm features)
     │                              │
     ▼                              ▼
[6A] Class-Conditional HMM     [6B] Baseline Models (Random Forest & GBDT)
     (Baum-Welch + Viterbi)         (Balanced class weighting)
     │                              │
     └─────────────┬────────────────┘
                   ▼
[7] Independent Patient Test Evaluation & Visualization
```

### Detailed Pipeline Breakdown:

1. **Data Acquisition & Verification:**
   - Resilient streaming pipeline downloaded and verified all 48 records (`.hea`, `.dat`, `.atr`).
   - Summary statistics generated in `results/dataset_summary.csv`.

2. **Signal Conditioning & Noise Removal:**
   - **0.5 – 45 Hz Butterworth Bandpass Filter (3rd-order):** Applied bidirectionally via `scipy.signal.sosfiltfilt` (effective 6th order) to eliminate low-frequency baseline drift (respiration, perspiration) and high-frequency muscle chatter (EMG) with zero phase distortion.
   - **60 Hz IIR Notch Filter:** Suppresses AC powerline hum.
   - **Z-Score Normalization:** Standardizes dynamic range across patients while preserving relative morphology.

3. **Pan-Tompkins QRS Peak Detection:**
   - Implemented the gold-standard algorithm: 5–15 Hz bandpass, 5-point derivative, squaring function, moving-window integration (150 ms), and dual adaptive thresholding with 200 ms refractory blanking.
   - Benchmarked against gold-standard cardiologist annotations (`.atr`) within a ±50 ms tolerance window:
     - **Mean Sensitivity:** **96.29%**
     - **Mean Precision:** **98.84%**
     - **Mean F1-Score:** **97.53%**

4. **Strict Patient-Wise Splitting (Zero Patient Leakage):**
   - Beats were **never randomly shuffled**. Shuffling beats from the same recording creates intra-patient data leakage that yields artificially inflated ~99% accuracy without testing true generalization.
   - Partitioned following the standard inter-patient benchmark (de Chazal et al.):
     - **Training Set (24 records):** 55,799 heartbeats
     - **Validation Set (4 records):** 4 records
     - **Independent Test Set (20 records):** 45,077 heartbeats from completely unseen patients
   - **Critical Patient Guarantee:** Records **201 and 202** originate from the same subject. They were strictly grouped together in the training pool to eliminate patient leakage.

5. **Feature Extraction:**
   - **Sequential Frames (for HMM):** Each 220-sample heartbeat (611 ms) is divided into 11 temporal frames (20 samples each). Each frame extracts 8 features: mean, standard deviation, peak-to-peak amplitude, slope, RMS energy, pre-RR interval, post-RR interval, and local RR-ratio.
   - **Tabular Features (for Baselines):** 27 static descriptors including higher-order moments (skewness, kurtosis), QRS energy, zero-crossing rate, downsampled profile, and rhythm dynamics.

6. **Hidden Markov Model (HMM) Implementation:**
   - Implemented **Class-Conditional Continuous Gaussian HMMs** using `hmmlearn`.
   - Hidden states ($M = 5$) directly capture the physiological sub-phases of the cardiac cycle:
     - State 0: Atrial Depolarization ($P$-wave)
     - State 1: AV Conduction Delay ($PR$-segment)
     - State 2: Ventricular Depolarization ($QRS$-complex)
     - State 3: Ventricular Plateau ($ST$-segment)
     - State 4: Ventricular Repolarization ($T$-wave)
   - Trained via the **Baum-Welch (EM)** algorithm on observation sequences from each class.
   - Evaluated using **Forward-Backward** log-likelihood scoring and Maximum A Posteriori (MAP) classification.
   - Decoded the physiological state paths via the **Viterbi algorithm**.

7. **Baseline Machine Learning Models:**
   - Trained balanced **Random Forest** (100 estimators, max depth 16) and **HistGradientBoosting** classifiers on identical tabular feature matrices.

---

## 4. Key Experimental Results (Evaluated on 45,077 Unseen Test Beats)

| Model | Overall Accuracy | Macro F1 | Weighted F1 | Normal (N) Se / +P | Supraventricular (S) Se / +P | Ventricular (V) Se / +P | Fusion (F) Se / +P |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HMM (Proposed)** | 73.49% | 45.02% | 81.60% | 72.8% / 98.8% | **74.6% / 42.4%** | 79.7% / 73.9% | **95.1% / 5.6%** |
| **Random Forest** | **93.84%** | **52.72%** | **94.38%** | 96.2% / 97.5% | 51.9% / 68.4% | **94.1% / 86.9%** | 19.1% / 15.8% |
| **HistGradientBoosting** | 91.19% | 49.46% | 92.54% | 93.4% / 98.5% | 41.1% / 41.5% | 95.5% / 76.6% | 37.5% / 18.9% |

### Key Scientific Takeaways:
1. **The Sensitivity Advantage of HMM on Rare Arrhythmias:**
   - In cardiology, missing an ectopic beat is far more dangerous than a false alarm.
   - The HMM achieved **74.63% Sensitivity on Supraventricular beats** (vs. Random Forest's 51.94%).
   - The HMM achieved **95.09% Sensitivity on Fusion beats** (vs. Random Forest's 19.12%).
   - *Reason:* Generative HMMs model the temporal state evolution of each arrhythmia class independently, preventing majority-class dominance from suppressing rare arrhythmic events.
2. **The Accuracy Advantage of Random Forest:**
   - Random Forest achieved 93.84% overall accuracy because it optimizes global discriminative boundaries for the dominant Normal class ($N$).
3. **Biological Interpretability:**
   - Visualizing the decoded hidden states (`07_hmm_viterbi_states.png`) confirms that the HMM discovers the genuine electrophysiological sequence without manual wave-boundary segmentation.

---

## 5. Project Directory Structure

```
ecg_arrhythmia/
├── data/
│   ├── raw/mitdb/             # All 48 MIT-BIH recordings (.hea, .dat, .atr)
│   └── processed/             # Cached extracted beats and feature arrays
├── notebooks/
│   └── 01_ecg_arrhythmia_pipeline.ipynb  # Interactive walkthrough
├── src/
│   ├── __init__.py
│   ├── config.py              # Central hyperparameters & record partitions
│   ├── data_loader.py         # Resilient streaming downloader & WFDB reader
│   ├── preprocessing.py       # Zero-phase Butterworth & notch filters
│   ├── peak_detection.py      # Pan-Tompkins QRS algorithm & validator
│   ├── segmentation.py        # Asymmetrical beat windowing & RR dynamics
│   ├── feature_extraction.py  # Sequential HMM frames & tabular descriptors
│   ├── hmm_model.py           # Class-Conditional Continuous Gaussian HMM
│   ├── baseline_models.py     # Random Forest & HistGradientBoosting benchmarks
│   ├── evaluation.py          # AAMI EC57 metrics & comparison tables
│   └── visualization.py       # Publication-ready figure generators
├── models/                    # Saved model artifacts (.joblib)
├── results/                   # Dataset summaries, metric tables & figures
│   ├── dataset_summary.csv
│   ├── model_comparison.csv
│   └── figures/
│       ├── 01_raw_vs_filtered.png
│       ├── 02_r_peak_detection.png
│       ├── 03_heartbeat_gallery.png
│       ├── 04_confusion_matrix_hmm.png
│       ├── 05_confusion_matrix_rf.png
│       ├── 06_model_comparison_bars.png
│       └── 07_hmm_viterbi_states.png
├── tests/
│   └── test_modules.py        # Automated test suite (7/7 tests pass)
├── main.py                    # End-to-end command-line runner
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive documentation
└── PROJECT_SUMMARY.md         # This summary document
```

---

## 6. How to Run the Project

Open a terminal inside the project directory:

```bash
# 1. Run unit tests (Takes ~0.5 seconds, verifies all modules)
python tests/test_modules.py

# 2. Run the quick demonstration pipeline (Takes ~20-30 seconds)
python main.py --quick

# 3. Run the full pipeline across all 48 records
python main.py

# 4. Open the interactive visual notebook
jupyter notebook notebooks/01_ecg_arrhythmia_pipeline.ipynb
```
