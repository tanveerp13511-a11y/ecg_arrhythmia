"""
Baseline Machine Learning Models for ECG Arrhythmia Classification.

Provides benchmark models to compare against the Hidden Markov Model (HMM):
1. Random Forest Classifier (balanced ensemble of decision trees)
2. Gradient Boosting / HistGradientBoosting Classifier (fast histogram-based tree boosting)
3. Support Vector Classifier (linear / RBF kernel)

All models are trained with class-weight balancing on the identical patient-split feature sets.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Allow running as a standalone script or module
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    AAMI_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    RANDOM_SEED,
    MODELS_DIR,
)


class BaselineClassifiers:
    """
    Container and trainer for baseline non-sequential machine learning classifiers.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        random_state: int = RANDOM_SEED
    ):
        """
        Parameters
        ----------
        model_type : str
            'random_forest' or 'hist_gbdt'.
        random_state : int
            Reproducibility seed.
        """
        self.model_type = model_type
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.classes = AAMI_CLASSES

    def fit(
        self,
        X_tabular: np.ndarray,
        y_labels: List[str]
    ) -> "BaselineClassifiers":
        """
        Train the baseline classifier on extracted tabular features.

        Parameters
        ----------
        X_tabular : np.ndarray
            2D feature matrix of shape (N_beats, N_features).
        y_labels : List[str]
            AAMI class strings ('N', 'S', 'V', 'F', 'Q').

        Returns
        -------
        BaselineClassifiers
            Self reference.
        """
        print(f"[Baseline] Training {self.model_type.upper()} classifier on {len(X_tabular):,} beats...")

        # Encode labels to integers
        y_encoded = np.array([CLASS_TO_IDX[label] for label in y_labels])

        if self.model_type == "random_forest":
            # Balanced class weights compensate for severe majority class (N) dominance
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=16,
                min_samples_split=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.random_state
            )
            # StandardScaler + Classifier
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", clf)
            ])
            
        elif self.model_type == "hist_gbdt":
            clf = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=8,
                class_weight="balanced",
                random_state=self.random_state
            )
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", clf)
            ])
        else:
            raise ValueError(f"Unknown baseline model type: {self.model_type}")

        self.pipeline.fit(X_tabular, y_encoded)
        print(f"[Baseline] {self.model_type.upper()} training complete.")
        return self

    def predict(self, X_tabular: np.ndarray) -> List[str]:
        """Predict AAMI class labels."""
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted before calling predict.")
        preds_int = self.pipeline.predict(X_tabular)
        return [IDX_TO_CLASS[idx] for idx in preds_int]

    def predict_proba(self, X_tabular: np.ndarray) -> np.ndarray:
        """Predict posterior probabilities."""
        if self.pipeline is None:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        return self.pipeline.predict_proba(X_tabular)

    def save(self, filepath: Path) -> None:
        """Serialize baseline model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[Baseline] Model saved to models/{filepath.name}")

    @classmethod
    def load(cls, filepath: Path) -> "BaselineClassifiers":
        """Deserialize baseline model."""
        filepath = Path(filepath)
        model = joblib.load(filepath)
        print(f"[Baseline] Model loaded from models/{filepath.name}")
        return model
