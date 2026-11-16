"""
R-Peak Detection Module for ECG Signals.

Implements:
1. The classic Pan-Tompkins QRS detection algorithm (Pan & Tompkins, IEEE TBME 1985).
2. Adaptive dual-thresholding with searchback and 200 ms physiological refractory blanking.
3. Fiducial alignment mapping back to original signal peaks.
4. Quantitative validation against gold-standard cardiologist annotations (.atr).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal as sp_signal

# Allow running as a standalone script or module
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    SAMPLING_RATE,
    R_PEAK_TOLERANCE_SAMPLES,
)


def pan_tompkins_qrs_detector(
    ecg_signal: np.ndarray,
    fs: float = SAMPLING_RATE
) -> np.ndarray:
    """
    Detect R-peaks in a single-lead ECG signal using the Pan-Tompkins algorithm.

    Pipeline Steps (Pan & Tompkins, 1985):
    1. Bandpass Filter (5 - 15 Hz): Isolates QRS complex energy, rejecting P/T waves & EMG.
    2. Five-point Derivative: Emphasizes steep slopes of the QRS complex.
    3. Squaring Function: Amplifies large QRS slopes nonlinearly and rectifies to positive values.
    4. Moving Window Integration (MWI): Extracts waveform energy envelope (width ~ 150 ms).
    5. Adaptive Dual-Thresholding & Searchback: Tracks dynamic signal and noise thresholds.
    6. Refractory Blanking: 200 ms window to prevent double-counting T-waves or split peaks.
    7. Fiducial Alignment: Refines detected locations to the exact local maximum in raw ECG.

    Parameters
    ----------
    ecg_signal : np.ndarray
        1D ECG signal array.
    fs : float
        Sampling frequency in Hz (default 360 Hz).

    Returns
    -------
    np.ndarray
        1D array of sample indices corresponding to detected R-peaks.
    """
    # Ensure 1D signal
    if ecg_signal.ndim > 1:
        signal_1d = ecg_signal[:, 0]
    else:
        signal_1d = ecg_signal

    n_samples = len(signal_1d)
    if n_samples < fs:
        return np.array([], dtype=int)

    # -------------------------------------------------------------
    # Step 1: Bandpass filter (5 - 15 Hz)
    # -------------------------------------------------------------
    nyquist = 0.5 * fs
    low = 5.0 / nyquist
    high = 15.0 / nyquist
    b_bp, a_bp = sp_signal.butter(1, [low, high], btype="bandpass")
    filtered_bp = sp_signal.filtfilt(b_bp, a_bp, signal_1d)

    # -------------------------------------------------------------
    # Step 2: Five-point derivative filter
    # Transfer function: H(z) = (1/8T) * (-z^-2 - 2z^-1 + 2z^1 + z^2)
    # -------------------------------------------------------------
    diff_kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) * (fs / 8.0)
    derivative = np.convolve(filtered_bp, diff_kernel, mode="same")

    # -------------------------------------------------------------
    # Step 3: Squaring function
    # -------------------------------------------------------------
    squared = derivative ** 2

    # -------------------------------------------------------------
    # Step 4: Moving Window Integration (MWI)
    # Window width of ~150 ms captures complete QRS complex duration
    # -------------------------------------------------------------
    window_len = int(0.150 * fs)
    mwi_kernel = np.ones(window_len) / window_len
    mwi = np.convolve(squared, mwi_kernel, mode="same")

    # -------------------------------------------------------------
    # Step 5: Adaptive Dual-Thresholding with Searchback
    # -------------------------------------------------------------
    # Find all local peaks in MWI signal
    min_distance = int(0.200 * fs)  # 200 ms physiological refractory limit
    peaks, properties = sp_signal.find_peaks(mwi, distance=min_distance)

    if len(peaks) == 0:
        return np.array([], dtype=int)

    # Initialize running signal and noise peak estimates
    first_peaks = peaks[:min(8, len(peaks))]
    spki = np.mean(mwi[first_peaks]) if len(first_peaks) > 0 else np.max(mwi) * 0.5
    npki = spki * 0.25

    detected_peaks_mwi = []
    rr_intervals = []

    for p in peaks:
        peak_val = mwi[p]
        threshold_i1 = npki + 0.25 * (spki - npki)

        # Check for searchback if no beat detected in 1.66 * average RR
        if detected_peaks_mwi and len(rr_intervals) >= 8:
            avg_rr = np.mean(rr_intervals[-8:])
            if (p - detected_peaks_mwi[-1]) > 1.66 * avg_rr:
                # Lower threshold for missed beats
                threshold_i2 = 0.5 * threshold_i1
                # Search candidate peaks in between
                missed = [cand for cand in peaks if detected_peaks_mwi[-1] < cand < p and mwi[cand] > threshold_i2]
                for m in missed:
                    detected_peaks_mwi.append(m)
                    if len(detected_peaks_mwi) >= 2:
                        rr_intervals.append(detected_peaks_mwi[-1] - detected_peaks_mwi[-2])

        # Evaluate current peak
        if peak_val >= threshold_i1:
            detected_peaks_mwi.append(p)
            spki = 0.125 * peak_val + 0.875 * spki
            if len(detected_peaks_mwi) >= 2:
                rr_intervals.append(detected_peaks_mwi[-1] - detected_peaks_mwi[-2])
        else:
            npki = 0.125 * peak_val + 0.875 * npki

    # -------------------------------------------------------------
    # Step 6: Fiducial Alignment
    # Map from MWI integration envelope to the true R-peak maximum
    # in the original ECG signal within a search window [-100ms, +50ms]
    # -------------------------------------------------------------
    search_pre = int(0.100 * fs)
    search_post = int(0.050 * fs)
    aligned_peaks = []

    for m_peak in detected_peaks_mwi:
        start_idx = max(0, m_peak - search_pre)
        end_idx = min(n_samples, m_peak + search_post)
        
        # Search for highest absolute peak (handles normal upright or inverted R-waves)
        window_signal = signal_1d[start_idx:end_idx]
        if len(window_signal) == 0:
            continue
        max_loc = np.argmax(np.abs(window_signal))
        true_r_peak = start_idx + max_loc
        
        # Enforce refractory period after alignment
        if not aligned_peaks or (true_r_peak - aligned_peaks[-1]) >= min_distance:
            aligned_peaks.append(true_r_peak)

    return np.array(aligned_peaks, dtype=int)


def evaluate_r_peak_detection(
    detected_peaks: np.ndarray,
    reference_peaks: np.ndarray,
    tolerance_samples: int = R_PEAK_TOLERANCE_SAMPLES
) -> Dict[str, float]:
    """
    Evaluate detected R-peaks against reference cardiologist annotations.

    In accordance with AAMI EC57 standards:
    - A detected peak is a True Positive (TP) if it falls within +/- tolerance
      (typically 50 ms = 18 samples at 360 Hz) of a reference beat.
    - An unmatched detected peak is a False Positive (FP).
    - An unmatched reference beat is a False Negative (FN).

    Parameters
    ----------
    detected_peaks : np.ndarray
        Array of detected R-peak sample indices.
    reference_peaks : np.ndarray
        Array of ground-truth reference sample indices.
    tolerance_samples : int
        Matching tolerance window in samples (default: +/- 18 samples for 50 ms).

    Returns
    -------
    Dict[str, float]
        Dictionary with TP, FP, FN, Sensitivity (Recall), Positive Predictivity (Precision), F1.
    """
    ref_matched = np.zeros(len(reference_peaks), dtype=bool)
    tp = 0
    fp = 0

    # Sort peaks for fast matching
    sorted_ref = np.sort(reference_peaks)
    sorted_det = np.sort(detected_peaks)

    for det in sorted_det:
        # Find reference peaks within tolerance
        diffs = np.abs(sorted_ref - det)
        within_tol_indices = np.where(diffs <= tolerance_samples)[0]

        # Match to the closest unmatched reference peak
        matched = False
        if len(within_tol_indices) > 0:
            # Sort within_tol by distance
            sorted_by_dist = within_tol_indices[np.argsort(diffs[within_tol_indices])]
            for candidate_idx in sorted_by_dist:
                if not ref_matched[candidate_idx]:
                    ref_matched[candidate_idx] = True
                    tp += 1
                    matched = True
                    break

        if not matched:
            fp += 1

    fn = len(sorted_ref) - tp

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * sensitivity * precision) / (sensitivity + precision) if (sensitivity + precision) > 0 else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "total_reference": len(sorted_ref),
        "total_detected": len(sorted_det),
        "Sensitivity_pct": round(sensitivity * 100.0, 2),
        "Precision_pct": round(precision * 100.0, 2),
        "F1_score_pct": round(f1 * 100.0, 2),
    }
