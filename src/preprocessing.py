"""
Signal Preprocessing Module for ECG Arrhythmia Detection.

This module provides signal conditioning algorithms to remove physiological
and electronic artifacts while preserving diagnostic ECG morphology:
1. Baseline wander removal (respiration, body movement): 0.5 Hz high-pass or two-stage median filter.
2. High-frequency noise suppression (EMG, electronic noise): 45 Hz low-pass filter.
3. Powerline interference rejection: 60 Hz notch filter.
4. Phase-distortion prevention: Zero-phase forward-backward filtering (scipy.signal.filtfilt).
5. Amplitude normalization: Z-score or robust median/IQR scaling.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from scipy import signal as sp_signal

# Allow running as a standalone script or module
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    SAMPLING_RATE,
    BANDPASS_LOWCUT,
    BANDPASS_HIGHCUT,
    FILTER_ORDER,
    NOTCH_FREQ,
    NOTCH_Q,
)


def butterworth_bandpass_filter(
    data: np.ndarray,
    lowcut: float = BANDPASS_LOWCUT,
    highcut: float = BANDPASS_HIGHCUT,
    fs: float = SAMPLING_RATE,
    order: int = FILTER_ORDER
) -> np.ndarray:
    """
    Apply a zero-phase bidirectional Butterworth bandpass filter.

    Clinical Rationale:
    - High-pass cutoff (0.5 Hz): Eliminates low-frequency baseline wander caused by
      patient respiration and perspiration, without distorting the ST segment.
    - Low-pass cutoff (45.0 Hz): Suppresses high-frequency electromyographic (EMG) muscle
      chatter and electronic noise, retaining the diagnostic QRS frequency band (10-30 Hz).
    - Zero-phase filtering (`filtfilt`): Applies the filter forward and backward, ensuring
      exactly zero phase shift so that R-peak locations and wave intervals remain unaltered.

    Parameters
    ----------
    data : np.ndarray
        1D or 2D ECG signal array.
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order (effective order is doubled by filtfilt).

    Returns
    -------
    np.ndarray
        Filtered ECG signal with identical shape and zero phase distortion.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design digital Butterworth bandpass filter coefficients (second-order sections)
    sos = sp_signal.butter(order, [low, high], btype="bandpass", output="sos")
    
    # Apply zero-phase forward-backward filtering
    if data.ndim == 1:
        filtered = sp_signal.sosfiltfilt(sos, data)
    else:
        filtered = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered[:, ch] = sp_signal.sosfiltfilt(sos, data[:, ch])
            
    return filtered


def notch_filter(
    data: np.ndarray,
    notch_freq: float = NOTCH_FREQ,
    fs: float = SAMPLING_RATE,
    Q: float = NOTCH_Q
) -> np.ndarray:
    """
    Apply a narrow notch filter to eliminate powerline hum (60 Hz in US MIT-BIH recordings).

    Parameters
    ----------
    data : np.ndarray
        ECG signal array.
    notch_freq : float
        Frequency to reject in Hz (default 60 Hz).
    fs : float
        Sampling rate in Hz.
    Q : float
        Quality factor defining notch sharpness (higher Q = narrower notch).

    Returns
    -------
    np.ndarray
        ECG signal with powerline frequency notched out.
    """
    b, a = sp_signal.iirnotch(notch_freq, Q, fs)
    
    if data.ndim == 1:
        return sp_signal.filtfilt(b, a, data)
    else:
        filtered = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered[:, ch] = sp_signal.filtfilt(b, a, data[:, ch])
        return filtered


def two_stage_median_filter(
    data: np.ndarray,
    fs: float = SAMPLING_RATE
) -> np.ndarray:
    """
    Remove baseline wander using two-stage median filtering (de Chazal et al. standard).

    How it works:
    1. First median filter (window ~ 200 ms = 71 samples at 360 Hz):
       Filters out QRS complexes and P-waves.
    2. Second median filter (window ~ 600 ms = 215 samples at 360 Hz):
       Filters out T-waves.
    3. The resulting smooth curve represents the pure baseline drift.
    4. Subtract the baseline drift from the original signal to yield baseline-corrected ECG.

    Parameters
    ----------
    data : np.ndarray
        1D ECG signal array.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Baseline-corrected ECG signal.
    """
    # Kernel sizes must be odd integers
    w1 = int(round(0.200 * fs))
    if w1 % 2 == 0:
        w1 += 1
    w2 = int(round(0.600 * fs))
    if w2 % 2 == 0:
        w2 += 1

    if data.ndim == 1:
        step1 = sp_signal.medfilt(data, kernel_size=w1)
        baseline = sp_signal.medfilt(step1, kernel_size=w2)
        return data - baseline
    else:
        corrected = np.zeros_like(data)
        for ch in range(data.shape[1]):
            step1 = sp_signal.medfilt(data[:, ch], kernel_size=w1)
            baseline = sp_signal.medfilt(step1, kernel_size=w2)
            corrected[:, ch] = data[:, ch] - baseline
        return corrected


def normalize_signal(
    data: np.ndarray,
    method: str = "zscore",
    robust: bool = False
) -> np.ndarray:
    """
    Normalize ECG signal amplitude.

    Methods:
    - 'zscore': Standardizes to mean=0, std=1 (recommended for HMM observation consistency).
    - 'minmax': Scales to [0, 1] or [-1, 1].
    - 'robust': Uses median and interquartile range (IQR) to avoid being distorted by pacing spikes.

    Parameters
    ----------
    data : np.ndarray
        ECG signal array.
    method : str
        'zscore' or 'minmax'.
    robust : bool
        If True, uses median & IQR for zscore scaling.

    Returns
    -------
    np.ndarray
        Normalized ECG signal.
    """
    eps = 1e-8
    if method == "zscore":
        if robust:
            median = np.median(data, axis=0)
            q75, q25 = np.percentile(data, [75, 25], axis=0)
            iqr = q75 - q25 + eps
            return (data - median) / iqr
        else:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0) + eps
            return (data - mean) / std
            
    elif method == "minmax":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + eps)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_ecg(
    raw_signal: np.ndarray,
    fs: float = SAMPLING_RATE,
    lowcut: float = BANDPASS_LOWCUT,
    highcut: float = BANDPASS_HIGHCUT,
    apply_notch: bool = True,
    norm_method: Optional[str] = "zscore"
) -> np.ndarray:
    """
    Complete end-to-end preprocessing pipeline for raw ECG signals.

    Workflow:
    1. Zero-phase Butterworth bandpass filter (0.5 - 45 Hz).
    2. Powerline notch filter at 60 Hz.
    3. Normalization (e.g. Z-score) per channel.

    Parameters
    ----------
    raw_signal : np.ndarray
        Raw ECG array in physical units (mV).
    fs : float
        Sampling frequency in Hz.
    lowcut : float
        Low cutoff frequency (0.5 Hz).
    highcut : float
        High cutoff frequency (45.0 Hz).
    apply_notch : bool
        Whether to apply 60 Hz powerline notch filtering.
    norm_method : Optional[str]
        'zscore', 'minmax', or None.

    Returns
    -------
    np.ndarray
        Preprocessed, clean ECG signal.
    """
    # 1. Bandpass filter
    filtered = butterworth_bandpass_filter(raw_signal, lowcut=lowcut, highcut=highcut, fs=fs)
    
    # 2. Notch filter (if sampling rate supports it)
    if apply_notch and (fs > 2 * NOTCH_FREQ):
        filtered = notch_filter(filtered, notch_freq=NOTCH_FREQ, fs=fs)
        
    # 3. Normalization
    if norm_method is not None:
        filtered = normalize_signal(filtered, method=norm_method)
        
    return filtered
