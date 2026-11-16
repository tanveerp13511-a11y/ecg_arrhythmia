"""
Heartbeat Segmentation Module for ECG Signals.

Extracts isolated cardiac cycles (heartbeats) centered around detected or reference R-peaks:
- Asymmetrical windowing (90 samples pre-R, 130 samples post-R = 220 samples ~ 611 ms at 360 Hz).
- Robust boundary condition handling (excludes or pads edge-clipped beats).
- Calculates RR interval dynamics (pre-RR, post-RR, local RR ratio).
- Mappings to standard AAMI 5-class labels (N, S, V, F, Q).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

# Allow running as a standalone script or module
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    SAMPLING_RATE,
    PRE_R_SAMPLES,
    POST_R_SAMPLES,
    WINDOW_SIZE,
    MIT_TO_AAMI,
    NON_BEAT_SYMBOLS,
    AAMI_CLASSES,
)


def extract_heartbeat_segments(
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    annotations: Optional[List[str]] = None,
    pre_r: int = PRE_R_SAMPLES,
    post_r: int = POST_R_SAMPLES,
    handle_boundary: str = "drop"
) -> Dict[str, Any]:
    """
    Segment a continuous ECG signal into individual heartbeat waveforms.

    Parameters
    ----------
    ecg_signal : np.ndarray
        1D ECG signal (e.g., Lead MLII) or 2D array.
    r_peaks : np.ndarray
        1D array of R-peak sample indices.
    annotations : Optional[List[str]]
        List of reference annotation symbols corresponding to each R-peak.
    pre_r : int
        Number of samples before R-peak (default: 90 samples = ~250 ms).
    post_r : int
        Number of samples after R-peak (default: 130 samples = ~361 ms).
    handle_boundary : str
        'drop': discard beats whose window falls outside signal boundaries.
        'pad': pad with edge values.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'segments': np.ndarray of shape (N_beats, WINDOW_SIZE)
        - 'r_peaks': np.ndarray of valid R-peak indices
        - 'symbols': list of original annotation symbols
        - 'aami_labels': list of mapped AAMI class strings
        - 'rr_pre': np.ndarray of preceding RR intervals in seconds
        - 'rr_post': np.ndarray of succeeding RR intervals in seconds
        - 'rr_ratio': np.ndarray of pre-RR divided by local 10-beat moving average RR
    """
    if ecg_signal.ndim > 1:
        signal_1d = ecg_signal[:, 0]
    else:
        signal_1d = ecg_signal

    n_samples = len(signal_1d)
    expected_len = pre_r + post_r

    valid_segments = []
    valid_r_peaks = []
    valid_symbols = []
    valid_aami = []
    valid_indices = []

    # Filter out non-beat annotations if provided
    has_annotations = annotations is not None and len(annotations) == len(r_peaks)

    for i, r in enumerate(r_peaks):
        sym = annotations[i] if has_annotations else "UNKNOWN"
        
        # Skip non-beat markers
        if sym in NON_BEAT_SYMBOLS:
            continue

        start = r - pre_r
        end = r + post_r

        # Boundary condition check
        if start < 0 or end > n_samples:
            if handle_boundary == "drop":
                continue
            elif handle_boundary == "pad":
                segment = np.zeros(expected_len, dtype=signal_1d.dtype)
                sig_start = max(0, start)
                sig_end = min(n_samples, end)
                pad_start = max(0, -start)
                pad_end = pad_start + (sig_end - sig_start)
                segment[pad_start:pad_end] = signal_1d[sig_start:sig_end]
            else:
                raise ValueError(f"Unknown boundary handler: {handle_boundary}")
        else:
            segment = signal_1d[start:end]

        valid_segments.append(segment)
        valid_r_peaks.append(r)
        valid_symbols.append(sym)
        valid_aami.append(MIT_TO_AAMI.get(sym, "OTHER"))
        valid_indices.append(i)

    if not valid_segments:
        return {
            "segments": np.empty((0, expected_len)),
            "r_peaks": np.array([]),
            "symbols": [],
            "aami_labels": [],
            "rr_pre": np.array([]),
            "rr_post": np.array([]),
            "rr_ratio": np.array([]),
        }

    valid_segments = np.array(valid_segments)
    valid_r_peaks = np.array(valid_r_peaks)

    # -------------------------------------------------------------
    # Calculate RR interval dynamics (in seconds)
    # -------------------------------------------------------------
    # Pre-RR interval: distance to previous beat
    rr_pre = np.zeros(len(valid_r_peaks))
    rr_pre[0] = (valid_r_peaks[1] - valid_r_peaks[0]) / SAMPLING_RATE if len(valid_r_peaks) > 1 else 0.8
    rr_pre[1:] = np.diff(valid_r_peaks) / SAMPLING_RATE

    # Post-RR interval: distance to next beat
    rr_post = np.zeros(len(valid_r_peaks))
    rr_post[:-1] = np.diff(valid_r_peaks) / SAMPLING_RATE
    rr_post[-1] = rr_pre[-1]

    # Local average RR ratio: pre-RR divided by 10-beat moving average RR
    # Premature beats (S and V) typically have sharp drops in RR ratio (< 0.85)
    rr_series = pd.Series(rr_pre)
    local_avg_rr = rr_series.rolling(window=10, min_periods=1, center=True).mean().values
    rr_ratio = rr_pre / (local_avg_rr + 1e-8)

    return {
        "segments": valid_segments,
        "r_peaks": valid_r_peaks,
        "symbols": valid_symbols,
        "aami_labels": valid_aami,
        "rr_pre": rr_pre,
        "rr_post": rr_post,
        "rr_ratio": rr_ratio,
    }


def segment_record_data(
    record_data: Dict[str, Any],
    preprocessed_signal: Optional[np.ndarray] = None,
    use_reference_peaks: bool = True,
    detected_peaks: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Convenience wrapper to segment a loaded MIT-BIH record.

    Parameters
    ----------
    record_data : Dict[str, Any]
        Output dictionary from `data_loader.load_record`.
    preprocessed_signal : Optional[np.ndarray]
        Cleaned ECG signal (if None, uses raw signal from record_data).
    use_reference_peaks : bool
        If True, uses cardiologist-annotated R-peaks from .atr file.
    detected_peaks : Optional[np.ndarray]
        If use_reference_peaks is False, uses these algorithm-detected peaks.

    Returns
    -------
    Dict[str, Any]
        Dictionary with extracted segments, annotations, and metadata.
    """
    sig = preprocessed_signal if preprocessed_signal is not None else record_data["signal"]
    
    if use_reference_peaks:
        peaks = record_data["ann_samples"]
        symbols = record_data["ann_symbols"]
    else:
        if detected_peaks is None:
            raise ValueError("detected_peaks must be provided when use_reference_peaks=False")
        peaks = detected_peaks
        symbols = None

    seg_dict = extract_heartbeat_segments(
        ecg_signal=sig,
        r_peaks=peaks,
        annotations=symbols,
    )
    seg_dict["record_id"] = record_data["record_id"]
    return seg_dict
