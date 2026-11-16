"""
Main Orchestration Pipeline for ECG Arrhythmia Detection and Classification.

Executes the complete research pipeline:
1. Dataset verification & acquisition (all 48 MIT-BIH records).
2. Database-wide summary generation and class distribution analysis.
3. Preprocessing (0.5 - 45 Hz bandpass, 60 Hz notch, z-score normalization).
4. Pan-Tompkins R-peak detection benchmark against reference annotations (.atr).
5. Heartbeat segmentation with strict inter-patient partitioning (zero patient leakage).
6. Sequential feature extraction (temporal frames) and tabular clinical features.
7. Hidden Markov Model (HMM) training using Baum-Welch (EM) algorithm.
8. Baseline model benchmarking (Random Forest & HistGradientBoosting).
9. ANSI/AAMI EC57 evaluation (Sensitivity, +Predictivity, Specificity, F1, Confusion Matrix).
10. Publication-quality figure generation.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    ALL_48_RECORDS,
    PATIENT_SPLIT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    SAMPLING_RATE,
    AAMI_CLASSES,
)
from src.data_loader import download_mitdb, verify_all_records, load_record, generate_dataset_summary
from src.preprocessing import preprocess_ecg
from src.peak_detection import pan_tompkins_qrs_detector, evaluate_r_peak_detection
from src.segmentation import extract_heartbeat_segments, segment_record_data
from src.feature_extraction import extract_sequential_features, extract_tabular_features
from src.hmm_model import ClassConditionalHMM
from src.baseline_models import BaselineClassifiers
from src.evaluation import compute_aami_metrics, format_metrics_table, compare_models
from src.visualization import (
    plot_raw_vs_filtered,
    plot_r_peak_detection,
    plot_heartbeat_gallery,
    plot_confusion_matrix,
    plot_model_comparison_bars,
    plot_hmm_viterbi_states,
)


def run_pipeline(
    max_train_beats_per_class: int = 5000,
    quick_mode: bool = False
) -> None:
    """
    Execute the complete ECG Arrhythmia detection and classification workflow.
    """
    start_time = time.time()
    print("=" * 80)
    print("  ECG ARRHYTHMIA DETECTION & CLASSIFICATION USING HIDDEN MARKOV MODELS (HMM)")
    print("  Complete MIT-BIH Arrhythmia Database (48 Recordings) - AAMI Standard")
    print("=" * 80)

    # -------------------------------------------------------------
    # Stage 1: Dataset Verification & Acquisition
    # -------------------------------------------------------------
    print("\n[STAGE 1/8] Verifying MIT-BIH Database Integrity (All 48 Recordings)...")
    complete, missing = verify_all_records(ALL_48_RECORDS, RAW_DATA_DIR)
    
    if len(missing) > 0:
        print(f"  --> {len(missing)} recordings are missing. Downloading from PhysioNet...")
        success_count, failed = download_mitdb(records=missing, dest_dir=RAW_DATA_DIR)
        if failed:
            print(f"  [ERROR] Could not download records: {failed}")
            sys.exit(1)
    else:
        print(f"  --> All {len(complete)} MIT-BIH recordings (.hea, .dat, .atr) verified on disk.")

    # -------------------------------------------------------------
    # Stage 2: Database Summary & Class Statistics
    # -------------------------------------------------------------
    print("\n[STAGE 2/8] Generating Clinical Dataset Summary...")
    summary_csv = RESULTS_DIR / "dataset_summary.csv"
    if summary_csv.exists():
        df_summary = pd.read_csv(summary_csv)
        print(f"  --> Loaded existing summary from results/dataset_summary.csv")
    else:
        df_summary = generate_dataset_summary(ALL_48_RECORDS, RAW_DATA_DIR, save_output=True)
    
    total_beats = df_summary["total_heartbeats"].sum()
    print(f"  --> Total Heartbeats Analyzed Across All 48 Records: {total_beats:,}")
    print("      Class Distribution:")
    for c in AAMI_CLASSES:
        c_beats = df_summary[c].sum()
        pct = (c_beats / total_beats) * 100.0
        print(f"        Class {c}: {c_beats:>7,} beats ({pct:>5.2f}%)")

    # -------------------------------------------------------------
    # Stage 3: Preprocessing & Filtering Demonstration
    # -------------------------------------------------------------
    print("\n[STAGE 3/8] Preprocessing ECG Signals & Noise Removal...")
    sample_rec_id = "100"
    rec100_data = load_record(sample_rec_id, RAW_DATA_DIR)
    raw_sig = rec100_data["signal"][:, 0]
    clean_sig = preprocess_ecg(raw_sig, fs=SAMPLING_RATE, norm_method="zscore")
    
    # Save visualization
    fig_filt_path = FIGURES_DIR / "01_raw_vs_filtered.png"
    plot_raw_vs_filtered(
        raw_signal=raw_sig,
        filtered_signal=clean_sig,
        fs=SAMPLING_RATE,
        duration_sec=10.0,
        start_sec=10.0,
        record_id=sample_rec_id,
        lead_name=rec100_data["lead_names"][0],
        save_path=fig_filt_path
    )

    # -------------------------------------------------------------
    # Stage 4: Pan-Tompkins QRS Peak Detection Benchmark
    # -------------------------------------------------------------
    print("\n[STAGE 4/8] Benchmarking Pan-Tompkins QRS Peak Detection...")
    # Evaluate across a representative subset of records
    eval_records = ["100", "101", "200", "201"] if quick_mode else ["100", "101", "103", "117", "200", "201", "203", "205"]
    qrs_results = []
    
    for rec in eval_records:
        rdata = load_record(rec, RAW_DATA_DIR)
        sig = rdata["signal"][:, 0]
        ref_peaks = rdata["ann_samples"]
        
        # Pan-Tompkins detection
        det_peaks = pan_tompkins_qrs_detector(sig, fs=SAMPLING_RATE)
        eval_metrics = evaluate_r_peak_detection(det_peaks, ref_peaks)
        qrs_results.append({
            "record": rec,
            "ref_beats": eval_metrics["total_reference"],
            "detected_beats": eval_metrics["total_detected"],
            "Sensitivity (%)": eval_metrics["Sensitivity_pct"],
            "Precision (%)": eval_metrics["Precision_pct"],
            "F1-Score (%)": eval_metrics["F1_score_pct"],
        })

    df_qrs = pd.DataFrame(qrs_results)
    print("  --> Pan-Tompkins QRS Detector Benchmark:")
    print(df_qrs.to_string(index=False))
    print(f"      Mean Sensitivity: {df_qrs['Sensitivity (%)'].mean():.2f}% | Mean F1-Score: {df_qrs['F1-Score (%)'].mean():.2f}%")

    # Save QRS detection plot on record 200 (frequent ventricular ectopy)
    r200_data = load_record("200", RAW_DATA_DIR)
    det_200 = pan_tompkins_qrs_detector(r200_data["signal"][:, 0], fs=SAMPLING_RATE)
    plot_r_peak_detection(
        ecg_signal=r200_data["signal"][:, 0],
        detected_peaks=det_200,
        reference_peaks=r200_data["ann_samples"],
        fs=SAMPLING_RATE,
        duration_sec=8.0,
        start_sec=15.0,
        record_id="200",
        save_path=FIGURES_DIR / "02_r_peak_detection.png"
    )

    # -------------------------------------------------------------
    # Stage 5: Patient-Wise Heartbeat Segmentation
    # -------------------------------------------------------------
    print("\n[STAGE 5/8] Segmenting Heartbeats with Strict Patient-Wise Partitioning...")
    print(f"  --> Train Records ({len(PATIENT_SPLIT['train'])}): {PATIENT_SPLIT['train']}")
    print(f"  --> Val Records   ({len(PATIENT_SPLIT['val'])}):   {PATIENT_SPLIT['val']}")
    print(f"  --> Test Records  ({len(PATIENT_SPLIT['test'])}):  {PATIENT_SPLIT['test']}")
    print("      *Notice: Records 201 & 202 are strictly grouped in the training pool (zero patient leakage).")

    def process_split_records(record_ids: List[str], desc: str) -> Dict[str, Any]:
        all_segments = []
        all_labels = []
        all_symbols = []
        all_rr_pre = []
        all_rr_post = []
        all_rr_ratio = []

        print(f"      Processing {desc} records ({len(record_ids)})...", flush=True)
        for rec in record_ids:
            rdata = load_record(rec, RAW_DATA_DIR)
            clean_s = preprocess_ecg(rdata["signal"][:, 0], fs=SAMPLING_RATE)
            seg_dict = segment_record_data(rdata, preprocessed_signal=clean_s)
            
            # Filter only valid AAMI classes
            valid_mask = np.isin(seg_dict["aami_labels"], AAMI_CLASSES)
            if np.sum(valid_mask) > 0:
                all_segments.append(seg_dict["segments"][valid_mask])
                all_labels.extend(np.array(seg_dict["aami_labels"])[valid_mask].tolist())
                all_symbols.extend(np.array(seg_dict["symbols"])[valid_mask].tolist())
                all_rr_pre.append(seg_dict["rr_pre"][valid_mask])
                all_rr_post.append(seg_dict["rr_post"][valid_mask])
                all_rr_ratio.append(seg_dict["rr_ratio"][valid_mask])

        return {
            "segments": np.vstack(all_segments),
            "labels": all_labels,
            "symbols": all_symbols,
            "rr_pre": np.concatenate(all_rr_pre),
            "rr_post": np.concatenate(all_rr_post),
            "rr_ratio": np.concatenate(all_rr_ratio),
        }

    train_data = process_split_records(PATIENT_SPLIT["train"], "TRAIN")
    test_data = process_split_records(PATIENT_SPLIT["test"], "TEST")

    print(f"  --> Segmented Beats: Train = {len(train_data['labels']):,} | Test = {len(test_data['labels']):,}")

    # Save Heartbeat Gallery plot
    plot_heartbeat_gallery(
        segments=train_data["segments"],
        aami_labels=train_data["labels"],
        samples_per_class=5,
        save_path=FIGURES_DIR / "03_heartbeat_gallery.png"
    )

    # -------------------------------------------------------------
    # Stage 6: Feature Extraction (Sequential for HMM & Tabular for Baselines)
    # -------------------------------------------------------------
    print("\n[STAGE 6/8] Extracting Sequential & Tabular Feature Representations...")
    
    # 1. Sequential features for HMM (11 frames x 8 features per beat)
    X_train_seq, lengths_train = extract_sequential_features(
        train_data["segments"], train_data["rr_pre"], train_data["rr_post"], train_data["rr_ratio"]
    )
    X_test_seq, lengths_test = extract_sequential_features(
        test_data["segments"], test_data["rr_pre"], test_data["rr_post"], test_data["rr_ratio"]
    )
    print(f"  --> Sequential HMM Matrix: Train = {X_train_seq.shape} | Test = {X_test_seq.shape}")

    # 2. Tabular features for Baseline ML (morphology, moments, QRS, RR dynamics)
    X_train_tab, feat_names = extract_tabular_features(
        train_data["segments"], train_data["rr_pre"], train_data["rr_post"], train_data["rr_ratio"]
    )
    X_test_tab, _ = extract_tabular_features(
        test_data["segments"], test_data["rr_pre"], test_data["rr_post"], test_data["rr_ratio"]
    )
    print(f"  --> Tabular Feature Matrix: Train = {X_train_tab.shape} | Test = {X_test_tab.shape} ({len(feat_names)} features)")

    # -------------------------------------------------------------
    # Stage 7: Model Training (HMM & Baselines)
    # -------------------------------------------------------------
    print("\n[STAGE 7/8] Training Hidden Markov Model & Baseline Classifiers...")
    
    # Model 1: Class-Conditional Gaussian HMM
    hmm_model = ClassConditionalHMM(
        n_states=5,
        n_frames=11,
        covariance_type="diag",
        n_iter=40,
        min_covar=1e-3,
        classes=AAMI_CLASSES
    )
    hmm_model.fit(
        X_train_seq,
        lengths_train,
        train_data["labels"],
        balanced_priors=True,
        max_samples_per_class=max_train_beats_per_class
    )
    hmm_model.save(MODELS_DIR / "hmm_class_conditional.joblib")

    # Model 2: Random Forest Baseline
    rf_model = BaselineClassifiers(model_type="random_forest")
    rf_model.fit(X_train_tab, train_data["labels"])
    rf_model.save(MODELS_DIR / "baseline_random_forest.joblib")

    # Model 3: Fast Histogram Gradient Boosting Baseline
    gbdt_model = BaselineClassifiers(model_type="hist_gbdt")
    gbdt_model.fit(X_train_tab, train_data["labels"])
    gbdt_model.save(MODELS_DIR / "baseline_hist_gbdt.joblib")

    # -------------------------------------------------------------
    # Stage 8: Evaluation on Independent Patient Test Set
    # -------------------------------------------------------------
    print("\n[STAGE 8/8] Evaluating Models on Independent Test Set (ANSI/AAMI EC57 Standard)...")
    
    # 1. HMM Evaluation
    print("  --> Evaluating Class-Conditional HMM...", flush=True)
    hmm_preds = hmm_model.predict(X_test_seq, lengths_test, use_priors=True)
    hmm_metrics = compute_aami_metrics(test_data["labels"], hmm_preds)

    # 2. Random Forest Evaluation
    print("  --> Evaluating Random Forest Classifier...", flush=True)
    rf_preds = rf_model.predict(X_test_tab)
    rf_metrics = compute_aami_metrics(test_data["labels"], rf_preds)

    # 3. GBDT Evaluation
    print("  --> Evaluating Gradient Boosting Classifier...", flush=True)
    gbdt_preds = gbdt_model.predict(X_test_tab)
    gbdt_metrics = compute_aami_metrics(test_data["labels"], gbdt_preds)

    # Display Per-Class HMM Results
    print("\n" + "=" * 70)
    print("  AAMI PERFORMANCE REPORT: HIDDEN MARKOV MODEL (HMM)")
    print("=" * 70)
    df_hmm_table = format_metrics_table(hmm_metrics)
    print(df_hmm_table.to_string(index=False))
    print(f"\n  Overall Accuracy: {hmm_metrics['overall_accuracy_pct']:.2f}% | Macro F1-Score: {hmm_metrics['macro_f1_pct']:.2f}%")

    # Side-by-Side Model Comparison Table
    print("\n" + "=" * 70)
    print("  SIDE-BY-SIDE MODEL BENCHMARK (Independent Patient Test Set)")
    print("=" * 70)
    all_results = {
        "HMM (Proposed)": hmm_metrics,
        "Random Forest": rf_metrics,
        "HistGradientBoosting": gbdt_metrics,
    }
    df_comp = compare_models(all_results, save_path=RESULTS_DIR / "model_comparison.csv")
    print(df_comp.to_string(index=False))

    # Save Confusion Matrices & Comparison Charts
    plot_confusion_matrix(
        cm=hmm_metrics["confusion_matrix"],
        classes=AAMI_CLASSES,
        model_name="Hidden Markov Model (HMM)",
        save_path=FIGURES_DIR / "04_confusion_matrix_hmm.png"
    )
    plot_confusion_matrix(
        cm=rf_metrics["confusion_matrix"],
        classes=AAMI_CLASSES,
        model_name="Random Forest Baseline",
        save_path=FIGURES_DIR / "05_confusion_matrix_rf.png"
    )
    plot_model_comparison_bars(
        comparison_df=df_comp,
        save_path=FIGURES_DIR / "06_model_comparison_bars.png"
    )

    # Viterbi Hidden State Decoding Visualization
    normal_indices = np.where(np.array(test_data["labels"]) == "N")[0]
    if len(normal_indices) > 0:
        sample_beat_idx = normal_indices[0]
        beat_seg = test_data["segments"][sample_beat_idx]
        beat_obs = X_test_seq[sample_beat_idx * 11 : (sample_beat_idx + 1) * 11]
        _, viterbi_states = hmm_model.decode_hidden_states(beat_obs, class_name="N")
        plot_hmm_viterbi_states(
            single_beat_segment=beat_seg,
            decoded_states=viterbi_states,
            class_name="N",
            save_path=FIGURES_DIR / "07_hmm_viterbi_states.png"
        )

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"  PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {elapsed/60.0:.2f} MINUTES")
    print(f"  Artifacts saved in: results/ and results/figures/")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Arrhythmia Classification with HMM")
    parser.add_argument("--max-train-beats", type=int, default=5000, help="Max training beats per class for HMM")
    parser.add_argument("--quick", action="store_true", help="Quick evaluation mode for rapid demonstration")
    args = parser.parse_args()

    run_pipeline(max_train_beats_per_class=args.max_train_beats, quick_mode=args.quick)
