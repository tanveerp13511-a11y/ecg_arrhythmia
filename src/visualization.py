"""
Publication-Quality Visualization Module for ECG Arrhythmia Analysis.

Provides plotting functions for:
1. Raw vs. Filtered ECG waveforms (demonstrating baseline wander & noise removal).
2. Pan-Tompkins detected R-peaks vs. Cardiologist reference annotations.
3. Heartbeat segmentation gallery across all 5 AAMI classes (N, S, V, F, Q).
4. Database-wide class distribution charts (log & linear scales).
5. Confusion matrices with count & percentage annotations.
6. Side-by-side comparative performance bar charts (HMM vs Baselines).
7. Hidden Markov Model Viterbi state path decoding over ECG morphology.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns

# Allow running as a standalone script or module
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    SAMPLING_RATE,
    AAMI_CLASSES,
    FIGURES_DIR,
)

# Apply consistent aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
})

# Standard clinical color palette for AAMI classes
AAMI_COLOR_MAP = {
    "N": "#2ecc71",  # Emerald Green (Normal)
    "S": "#f39c12",  # Orange (Supraventricular)
    "V": "#e74c3c",  # Crimson Red (Ventricular)
    "F": "#9b59b6",  # Amethyst Purple (Fusion)
    "Q": "#34495e",  # Dark Slate (Paced / Unknown)
}


def plot_raw_vs_filtered(
    raw_signal: np.ndarray,
    filtered_signal: np.ndarray,
    fs: float = SAMPLING_RATE,
    duration_sec: float = 10.0,
    start_sec: float = 0.0,
    record_id: str = "100",
    lead_name: str = "MLII",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot raw vs zero-phase filtered ECG signal to illustrate noise & baseline removal.
    """
    start_sample = int(start_sec * fs)
    n_samples = int(duration_sec * fs)
    end_sample = min(len(raw_signal), start_sample + n_samples)
    time_axis = np.arange(start_sample, end_sample) / fs

    sig_raw = raw_signal[start_sample:end_sample]
    sig_filt = filtered_signal[start_sample:end_sample]
    if sig_raw.ndim > 1:
        sig_raw = sig_raw[:, 0]
    if sig_filt.ndim > 1:
        sig_filt = sig_filt[:, 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, dpi=300)

    # 1. Raw ECG
    ax1.plot(time_axis, sig_raw, color="#c0392b", linewidth=1.1, label="Raw ECG (Lead " + lead_name + ")")
    ax1.set_title(f"MIT-BIH Record {record_id} - Raw Unfiltered Signal (Showing Baseline Drift & Powerline Noise)", fontweight="bold")
    ax1.set_ylabel("Amplitude (mV)")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Filtered ECG
    ax2.plot(time_axis, sig_filt, color="#2980b9", linewidth=1.1, label="Filtered ECG (0.5 - 45 Hz Zero-Phase Bandpass + 60 Hz Notch)")
    ax2.set_title("Preprocessed Signal (Baseline Wander Removed, QRS Complexes & ST-T Morphology Preserved)", fontweight="bold")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Normalized Amplitude")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Visualization] Saved raw vs filtered plot to results/figures/{Path(save_path).name}")

    return fig


def plot_r_peak_detection(
    ecg_signal: np.ndarray,
    detected_peaks: np.ndarray,
    reference_peaks: np.ndarray,
    fs: float = SAMPLING_RATE,
    duration_sec: float = 10.0,
    start_sec: float = 0.0,
    record_id: str = "100",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot detected Pan-Tompkins R-peaks alongside reference cardiologist annotations.
    """
    start_sample = int(start_sec * fs)
    end_sample = min(len(ecg_signal), start_sample + int(duration_sec * fs))
    time_axis = np.arange(start_sample, end_sample) / fs

    sig = ecg_signal[start_sample:end_sample]
    if sig.ndim > 1:
        sig = sig[:, 0]

    # Filter peaks within window
    det_in_window = detected_peaks[(detected_peaks >= start_sample) & (detected_peaks < end_sample)]
    ref_in_window = reference_peaks[(reference_peaks >= start_sample) & (reference_peaks < end_sample)]

    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    ax.plot(time_axis, sig, color="#2c3e50", linewidth=1.2, label="ECG Signal")

    # Reference beats (Green circles)
    if len(ref_in_window) > 0:
        ax.scatter(
            ref_in_window / fs,
            sig[ref_in_window - start_sample],
            color="#27ae60",
            s=80,
            marker="o",
            facecolors="none",
            edgecolors="#27ae60",
            linewidths=2.0,
            label="PhysioNet Reference Annotation (.atr)",
            zorder=4
        )

    # Detected peaks (Red triangles)
    if len(det_in_window) > 0:
        ax.scatter(
            det_in_window / fs,
            sig[det_in_window - start_sample],
            color="#e74c3c",
            s=70,
            marker="^",
            label="Pan-Tompkins Detected R-Peak",
            zorder=5
        )

    ax.set_title(f"R-Peak Detection Evaluation (MIT-BIH Record {record_id})", fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Visualization] Saved R-peak plot to results/figures/{Path(save_path).name}")

    return fig


def plot_heartbeat_gallery(
    segments: np.ndarray,
    aami_labels: List[str],
    samples_per_class: int = 5,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot representative segmented heartbeats for each AAMI superclass (N, S, V, F, Q).
    """
    labels_arr = np.array(aami_labels)
    unique_classes = [c for c in AAMI_CLASSES if c in labels_arr]
    
    n_rows = len(unique_classes)
    fig, axes = plt.subplots(n_rows, samples_per_class, figsize=(15, 2.5 * n_rows), sharex=True, sharey=True, dpi=300)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    time_samples = np.arange(segments.shape[1]) - 90  # Centered at R-peak = sample 0

    for r_idx, c in enumerate(unique_classes):
        indices = np.where(labels_arr == c)[0]
        selected = indices[:min(samples_per_class, len(indices))]
        color = AAMI_COLOR_MAP.get(c, "#34495e")

        for col_idx in range(samples_per_class):
            ax = axes[r_idx, col_idx]
            if col_idx < len(selected):
                ax.plot(time_samples, segments[selected[col_idx]], color=color, linewidth=1.5)
                ax.axvline(0, color="gray", linestyle=":", alpha=0.7)  # R-peak center
            else:
                ax.set_visible(False)

            if col_idx == 0:
                ax.set_ylabel(f"Class {c}", fontweight="bold", fontsize=12)
            if r_idx == n_rows - 1:
                ax.set_xlabel("Samples relative to R-peak")
            ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Representative Segmented Heartbeats by AAMI Arrhythmia Class (Centered at R-Peak)", fontweight="bold", y=0.99)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Visualization] Saved heartbeat gallery to results/figures/{Path(save_path).name}")

    return fig


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    classes: List[str] = AAMI_CLASSES,
    model_name: str = "HMM",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot a standardized clinical confusion matrix with both count and row-normalized recall %.
    """
    cm_arr = np.array(cm)
    cm_norm = cm_arr.astype("float") / (cm_arr.sum(axis=1)[:, np.newaxis] + 1e-8) * 100.0

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    # Create combined annotation text (Count \n (Recall %))
    annot_matrix = []
    for i in range(len(classes)):
        row = []
        for j in range(len(classes)):
            count = cm_arr[i, j]
            pct = cm_norm[i, j]
            row.append(f"{count:,}\n({pct:.1f}%)")
        annot_matrix.append(row)

    sns.heatmap(
        cm_norm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Normalized Recall (% per True Class)"},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        vmin=0,
        vmax=100
    )

    ax.set_title(f"AAMI Heartbeat Confusion Matrix - {model_name} Model", fontweight="bold", pad=15)
    ax.set_xlabel("Predicted AAMI Class", fontweight="bold", labelpad=10)
    ax.set_ylabel("True Reference AAMI Class", fontweight="bold", labelpad=10)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Visualization] Saved confusion matrix to results/figures/{save_path.name}")

    return fig


def plot_model_comparison_bars(
    comparison_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot side-by-side bar chart comparing overall accuracy and macro F1 across models.
    """
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    # Clean numerical columns
    models = comparison_df["Model"].tolist()
    acc_vals = [float(x.replace("%", "")) for x in comparison_df["Overall Acc (%)"]]
    f1_vals = [float(x.replace("%", "")) for x in comparison_df["Macro F1 (%)"]]

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax.bar(x - width/2, acc_vals, width, label="Overall Accuracy (%)", color="#3498db")
    rects2 = ax.bar(x + width/2, f1_vals, width, label="Macro F1-Score (%)", color="#e67e22")

    # Add labels on bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title("Performance Benchmark: Hidden Markov Model vs. Baseline ML Models", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Visualization] Saved model comparison chart to results/figures/{save_path.name}")

    return fig


def plot_hmm_viterbi_states(
    single_beat_segment: np.ndarray,
    decoded_states: np.ndarray,
    class_name: str = "N",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Visualize how the HMM's Viterbi hidden states correspond to the cardiac cycle sub-phases.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    
    n_samples = len(single_beat_segment)
    n_frames = len(decoded_states)
    frame_len = n_samples // n_frames
    
    state_names = {
        0: "State 0: P-Wave (Atrial Depol.)",
        1: "State 1: PR-Segment (AV Delay)",
        2: "State 2: QRS (Ventricular Depol.)",
        3: "State 3: ST-Segment (Plateau)",
        4: "State 4: T-Wave (Repol.)"
    }
    
    colors = ["#3498db", "#f39c12", "#e74c3c", "#9b59b6", "#2ecc71"]
    time_pts = np.arange(n_samples) - 90  # Centered at R-peak
    
    # Plot base waveform
    ax.plot(time_pts, single_beat_segment, color="#2c3e50", linewidth=1.5, label="ECG Waveform")

    # Highlight background according to Viterbi decoded state
    for f_idx, state in enumerate(decoded_states):
        s_start = f_idx * frame_len - 90
        s_end = (f_idx + 1) * frame_len - 90 if f_idx < n_frames - 1 else n_samples - 90
        color = colors[state % len(colors)]
        ax.axvspan(s_start, s_end, color=color, alpha=0.25)

    ax.axvline(0, color="black", linestyle="--", alpha=0.7, label="R-Peak")
    ax.set_title(f"Hidden Markov Model Viterbi State Decoding (Class {class_name} Beat)", fontweight="bold")
    ax.set_xlabel("Samples relative to R-peak")
    ax.set_ylabel("Normalized Amplitude")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Visualization] Saved Viterbi state plot to results/figures/{save_path.name}")

    return fig
