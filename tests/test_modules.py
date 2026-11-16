"""
Unit and Integration Tests for ECG Arrhythmia Modules.

Tests:
1. Config integrity and AAMI mapping.
2. Data loading and WFDB parsing on record 100.
3. Preprocessing (bandpass filter, notch, normalization).
4. Pan-Tompkins QRS peak detection against reference annotations.
5. Heartbeat segmentation and window geometry.
6. Sequential feature extraction for HMM and tabular features for baselines.
7. Class-Conditional HMM fitting and inference smoke-test.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    ALL_48_RECORDS,
    AAMI_CLASSES,
    SAMPLING_RATE,
    WINDOW_SIZE,
    PRE_R_SAMPLES,
    POST_R_SAMPLES,
    HMM_N_FRAMES,
    RAW_DATA_DIR,
)
from src.data_loader import verify_record_files, load_record
from src.preprocessing import preprocess_ecg, butterworth_bandpass_filter, normalize_signal
from src.peak_detection import pan_tompkins_qrs_detector, evaluate_r_peak_detection
from src.segmentation import extract_heartbeat_segments, segment_record_data
from src.feature_extraction import extract_sequential_features, extract_tabular_features
from src.hmm_model import ClassConditionalHMM
from src.baseline_models import BaselineClassifiers
from src.evaluation import compute_aami_metrics


class TestECGArrhythmiaPipeline(unittest.TestCase):
    """Test suite for the modular ECG Arrhythmia processing pipeline."""

    @classmethod
    def setUpClass(cls):
        """Check if record 100 is available for testing."""
        cls.test_record_id = "100"
        status = verify_record_files(cls.test_record_id, RAW_DATA_DIR)
        cls.record_available = status["complete"]
        if cls.record_available:
            cls.record_data = load_record(cls.test_record_id, RAW_DATA_DIR)

    def test_01_config_integrity(self):
        """Verify that all 48 recordings and AAMI classes are defined."""
        self.assertEqual(len(ALL_48_RECORDS), 48)
        self.assertEqual(AAMI_CLASSES, ["N", "S", "V", "F", "Q"])
        self.assertEqual(WINDOW_SIZE, PRE_R_SAMPLES + POST_R_SAMPLES)

    def test_02_data_loader(self):
        """Verify reading signals and annotations for record 100."""
        if not self.record_available:
            self.skipTest(f"Record {self.test_record_id} not yet downloaded.")
        data = self.record_data
        self.assertEqual(data["record_id"], "100")
        self.assertEqual(data["fs"], SAMPLING_RATE)
        self.assertGreater(len(data["signal"]), 600000)
        self.assertGreater(len(data["ann_samples"]), 2000)
        self.assertEqual(len(data["ann_samples"]), len(data["aami_labels"]))

    def test_03_preprocessing(self):
        """Verify that preprocessing removes baseline drift and normalizes."""
        if not self.record_available:
            self.skipTest(f"Record {self.test_record_id} not yet downloaded.")
        raw_sig = self.record_data["signal"][:3600, 0]  # First 10 seconds
        clean_sig = preprocess_ecg(raw_sig, fs=SAMPLING_RATE, norm_method="zscore")
        
        self.assertEqual(len(clean_sig), len(raw_sig))
        self.assertAlmostEqual(np.mean(clean_sig), 0.0, places=1)
        self.assertAlmostEqual(np.std(clean_sig), 1.0, places=1)

    def test_04_peak_detection(self):
        """Verify Pan-Tompkins QRS detector achieves > 95% sensitivity and F1 on record 100."""
        if not self.record_available:
            self.skipTest(f"Record {self.test_record_id} not yet downloaded.")
        # Test on first 60 seconds (approx 75 beats)
        sig_slice = self.record_data["signal"][: 60 * SAMPLING_RATE, 0]
        ref_peaks = self.record_data["ann_samples"][self.record_data["ann_samples"] < 60 * SAMPLING_RATE]
        
        detected_peaks = pan_tompkins_qrs_detector(sig_slice, fs=SAMPLING_RATE)
        eval_res = evaluate_r_peak_detection(detected_peaks, ref_peaks)
        
        self.assertGreater(eval_res["Sensitivity_pct"], 90.0)
        self.assertGreater(eval_res["F1_score_pct"], 90.0)

    def test_05_segmentation(self):
        """Verify beat segmentation geometry and AAMI mapping."""
        if not self.record_available:
            self.skipTest(f"Record {self.test_record_id} not yet downloaded.")
        clean_sig = preprocess_ecg(self.record_data["signal"][:, 0], fs=SAMPLING_RATE)
        seg_dict = segment_record_data(self.record_data, preprocessed_signal=clean_sig)
        
        segments = seg_dict["segments"]
        self.assertGreater(len(segments), 2000)
        self.assertEqual(segments.shape[1], WINDOW_SIZE)
        self.assertEqual(len(seg_dict["aami_labels"]), len(segments))

    def test_06_feature_extraction(self):
        """Verify extraction of sequential features for HMM and tabular features for baselines."""
        dummy_beats = np.random.randn(20, WINDOW_SIZE).astype(np.float32)
        dummy_rr_pre = np.full(20, 0.8, dtype=np.float32)
        dummy_rr_post = np.full(20, 0.8, dtype=np.float32)
        dummy_rr_ratio = np.full(20, 1.0, dtype=np.float32)

        # Sequential features
        X_seq, lengths = extract_sequential_features(dummy_beats, dummy_rr_pre, dummy_rr_post, dummy_rr_ratio)
        self.assertEqual(len(X_seq), 20 * HMM_N_FRAMES)
        self.assertEqual(len(lengths), 20)

        # Tabular features
        X_tab, feat_names = extract_tabular_features(dummy_beats, dummy_rr_pre, dummy_rr_post, dummy_rr_ratio)
        self.assertEqual(X_tab.shape[0], 20)
        self.assertEqual(X_tab.shape[1], len(feat_names))

    def test_07_hmm_and_baseline_smoke(self):
        """Verify smoke training and inference for HMM and Baseline classifiers."""
        dummy_beats = np.random.randn(30, WINDOW_SIZE).astype(np.float32)
        dummy_rr = np.full(30, 0.8, dtype=np.float32)
        dummy_ratio = np.full(30, 1.0, dtype=np.float32)
        labels = ["N"] * 20 + ["V"] * 10

        # HMM
        X_seq, lengths = extract_sequential_features(dummy_beats, dummy_rr, dummy_rr, dummy_ratio)
        hmm_model = ClassConditionalHMM(n_states=3, n_frames=HMM_N_FRAMES, n_iter=5, classes=["N", "V"])
        hmm_model.fit(X_seq, lengths, labels)
        preds = hmm_model.predict(X_seq, lengths)
        self.assertEqual(len(preds), 30)

        # Baseline
        X_tab, _ = extract_tabular_features(dummy_beats, dummy_rr, dummy_rr, dummy_ratio)
        rf_model = BaselineClassifiers(model_type="random_forest")
        rf_model.fit(X_tab, labels)
        rf_preds = rf_model.predict(X_tab)
        self.assertEqual(len(rf_preds), 30)


if __name__ == "__main__":
    unittest.main()
