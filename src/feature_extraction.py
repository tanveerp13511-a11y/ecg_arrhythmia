"""
Feature Extraction Module for ECG Heartbeats.

Constructs two complementary feature sets:
1. Sequential Observation Vectors for HMMs:
   - Divides each 220-sample heartbeat window into temporal sub-frames (e.g. 11 frames of 20 samples).
   - Extracts frame-level morphological descriptors (mean, std, energy, slope, peak-to-peak).
   - Enables Hidden Markov Models to track the temporal transition across cardiac phases
     (P-wave -> PR segment -> QRS depolarization -> ST segment -> T-wave repolarization).

2. Global Tabular Feature Vectors for Baseline Machine Learning Models (RF, SVM, GBDT):
   - Rhythm dynamics: Pre-RR, Post-RR, Local RR ratio.
   - Higher-order statistical moments: Mean, variance, skewness, kurtosis.
   - Morphological and waveform energy descriptors.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from scipy import stats

# Allow running as a standalone script or module
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    WINDOW_SIZE,
    HMM_N_FRAMES,
    SAMPLING_RATE,
)


def extract_sequential_features(
    beat_segments: np.ndarray,
    rr_pre: np.ndarray,
    rr_post: np.ndarray,
    rr_ratio: np.ndarray,
    n_frames: int = HMM_N_FRAMES,
    include_rr_context: bool = True
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract sequential observation matrices for Hidden Markov Models (HMM).

    Each heartbeat segment (length L = 220 samples) is partitioned into
    `n_frames` contiguous temporal sub-frames (frame_len = L // n_frames = 20 samples).
    
    For each frame, we compute:
    1. Mean amplitude
    2. Standard deviation
    3. Peak-to-peak amplitude (max - min)
    4. First derivative / slope across frame
    5. Root Mean Square (RMS) energy
    6. (Optional) RR interval context (pre-RR, post-RR, ratio) appended per frame.

    Parameters
    ----------
    beat_segments : np.ndarray
        Array of heartbeat waveforms, shape (N_beats, WINDOW_SIZE).
    rr_pre : np.ndarray
        Preceding RR intervals in seconds, shape (N_beats,).
    rr_post : np.ndarray
        Succeeding RR intervals in seconds, shape (N_beats,).
    rr_ratio : np.ndarray
        Pre-RR divided by local moving average RR, shape (N_beats,).
    n_frames : int
        Number of sequential frames per heartbeat (default: 11 frames of 20 samples).
    include_rr_context : bool
        Whether to append RR interval features to each temporal frame.

    Returns
    -------
    Tuple[np.ndarray, List[int]]
        - X_seq: 2D array of concatenated observations of shape (N_beats * n_frames, n_features),
          ready for hmmlearn's `fit(X, lengths)`.
        - lengths: List of length N_beats, each element equal to `n_frames`.
    """
    n_beats, sig_len = beat_segments.shape
    frame_len = sig_len // n_frames
    
    # Base frame features: [mean, std, peak_to_peak, slope, rms] -> 5 features
    # If include_rr_context: +3 features [rr_pre, rr_post, rr_ratio] -> 8 features
    n_feats = 8 if include_rr_context else 5
    
    # 3D array: (N_beats, n_frames, n_feats)
    seq_matrix = np.zeros((n_beats, n_frames, n_feats), dtype=np.float32)

    for f_idx in range(n_frames):
        start = f_idx * frame_len
        end = start + frame_len if f_idx < n_frames - 1 else sig_len
        frame = beat_segments[:, start:end]

        # 1. Mean amplitude in frame
        seq_matrix[:, f_idx, 0] = np.mean(frame, axis=1)
        # 2. Standard deviation in frame
        seq_matrix[:, f_idx, 1] = np.std(frame, axis=1)
        # 3. Peak-to-peak amplitude
        seq_matrix[:, f_idx, 2] = np.ptp(frame, axis=1)
        # 4. Slope across frame (end - start)
        seq_matrix[:, f_idx, 3] = (frame[:, -1] - frame[:, 0]) / frame_len
        # 5. RMS energy
        seq_matrix[:, f_idx, 4] = np.sqrt(np.mean(frame ** 2, axis=1))

        if include_rr_context:
            seq_matrix[:, f_idx, 5] = rr_pre
            seq_matrix[:, f_idx, 6] = rr_post
            seq_matrix[:, f_idx, 7] = rr_ratio

    # Reshape for hmmlearn: (N_beats * n_frames, n_feats)
    X_seq = seq_matrix.reshape(n_beats * n_frames, n_feats)
    lengths = [n_frames] * n_beats

    return X_seq, lengths


def extract_tabular_features(
    beat_segments: np.ndarray,
    rr_pre: np.ndarray,
    rr_post: np.ndarray,
    rr_ratio: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract comprehensive tabular features per beat for baseline classifiers (RF, SVM).

    Feature groups:
    1. Rhythm Features (3):
       - pre_rr, post_rr, rr_ratio
    2. Statistical Moments (4):
       - mean, variance, skewness, kurtosis
    3. Waveform Morphology & Extremes (7):
       - min, max, peak-to-peak, argmin, argmax, zero-crossings, energy
    4. QRS Complex Characteristics (4):
       - QRS peak amplitude, QRS energy, QRS slope, post-R/pre-R amplitude ratio
    5. Subband Downsampled Profile (11):
       - Mean amplitude across 11 sub-segments (captures overall wave shape)

    Parameters
    ----------
    beat_segments : np.ndarray
        Array of heartbeat waveforms, shape (N_beats, WINDOW_SIZE).
    rr_pre : np.ndarray
        Preceding RR intervals in seconds.
    rr_post : np.ndarray
        Succeeding RR intervals in seconds.
    rr_ratio : np.ndarray
        Pre-RR divided by local moving average RR.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        - X_tabular: 2D feature matrix of shape (N_beats, N_features).
        - feature_names: List of feature names.
    """
    n_beats, sig_len = beat_segments.shape
    features_list = []
    feature_names = []

    # 1. Rhythm features
    features_list.append(rr_pre[:, None])
    features_list.append(rr_post[:, None])
    features_list.append(rr_ratio[:, None])
    feature_names.extend(["pre_rr", "post_rr", "rr_ratio"])

    # 2. Statistical moments
    mean_val = np.mean(beat_segments, axis=1, keepdims=True)
    var_val = np.var(beat_segments, axis=1, keepdims=True)
    skew_val = stats.skew(beat_segments, axis=1)[:, None]
    kurt_val = stats.kurtosis(beat_segments, axis=1)[:, None]
    features_list.extend([mean_val, var_val, skew_val, kurt_val])
    feature_names.extend(["mean", "var", "skewness", "kurtosis"])

    # 3. Waveform morphology & extremes
    min_val = np.min(beat_segments, axis=1, keepdims=True)
    max_val = np.max(beat_segments, axis=1, keepdims=True)
    ptp_val = max_val - min_val
    argmin_val = (np.argmin(beat_segments, axis=1, keepdims=True) / sig_len)
    argmax_val = (np.argmax(beat_segments, axis=1, keepdims=True) / sig_len)
    energy_val = np.sum(beat_segments ** 2, axis=1, keepdims=True) / sig_len
    features_list.extend([min_val, max_val, ptp_val, argmin_val, argmax_val, energy_val])
    feature_names.extend(["min", "max", "ptp", "argmin_norm", "argmax_norm", "energy"])

    # Zero-crossing rate
    zcr = np.mean(np.diff(np.sign(beat_segments) != 0, axis=1), axis=1, keepdims=True)
    features_list.append(zcr)
    feature_names.append("zero_crossings")

    # 4. QRS segment characteristics (window [70:110] centered around R-peak sample 90)
    qrs_window = beat_segments[:, 70:110]
    qrs_energy = np.sum(qrs_window ** 2, axis=1, keepdims=True) / 40.0
    r_peak_amp = beat_segments[:, 90:91]
    features_list.extend([r_peak_amp, qrs_energy])
    feature_names.extend(["r_peak_amplitude", "qrs_energy"])

    # 5. Downsampled temporal profile (11 sub-segments)
    n_sub = 11
    sub_len = sig_len // n_sub
    for s in range(n_sub):
        sub_mean = np.mean(beat_segments[:, s*sub_len:(s+1)*sub_len], axis=1, keepdims=True)
        features_list.append(sub_mean)
        feature_names.append(f"profile_sub_{s}")

    X_tabular = np.hstack(features_list).astype(np.float32)
    return X_tabular, feature_names
