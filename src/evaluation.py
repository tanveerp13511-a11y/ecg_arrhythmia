"""
Model Evaluation Module for AAMI Arrhythmia Heartbeat Classification.

Computes:
1. ANSI/AAMI EC57 clinical performance metrics:
   - Sensitivity (Recall) per class: Se = TP / (TP + FN)
   - Positive Predictivity (Precision) per class: +P = TP / (TP + FP)
   - Specificity per class: Sp = TN / (TN + FP)
   - Per-class F1-Score
2. Overall Accuracy and Macro-averaged F1-Score (essential for severe class imbalance).
3. Confusion Matrix (rows = true class, columns = predicted class).
4. Side-by-side model comparison tables (HMM vs Baseline ML models).
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# Allow running as a standalone script or module
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    AAMI_CLASSES,
    CLASS_TO_IDX,
    RESULTS_DIR,
)


def compute_aami_metrics(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str] = AAMI_CLASSES
) -> Dict[str, Any]:
    """
    Compute comprehensive clinical metrics adhering to the ANSI/AAMI EC57 standard.

    Parameters
    ----------
    y_true : List[str]
        Ground-truth AAMI class labels.
    y_pred : List[str]
        Predicted AAMI class labels.
    classes : List[str]
        Class label ordering (default: ['N', 'S', 'V', 'F', 'Q']).

    Returns
    -------
    Dict[str, Any]
        Dictionary with per-class and overall metrics plus raw confusion matrix.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Compute confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=classes)
    total_samples = len(y_true_arr)
    overall_acc = accuracy_score(y_true_arr, y_pred_arr)
    macro_f1 = f1_score(y_true_arr, y_pred_arr, labels=classes, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_arr, y_pred_arr, labels=classes, average="weighted", zero_division=0)

    per_class_metrics = {}
    for idx, c in enumerate(classes):
        tp = int(cm[idx, idx])
        fn = int(np.sum(cm[idx, :]) - tp)
        fp = int(np.sum(cm[:, idx]) - tp)
        tn = int(total_samples - (tp + fn + fp))

        se = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0  # Sensitivity (Recall)
        pp = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0  # Positive Predictivity (Precision)
        sp = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 0.0  # Specificity
        f1 = (2 * se * pp) / (se + pp) if (se + pp) > 0 else 0.0

        per_class_metrics[c] = {
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
            "Support": tp + fn,
            "Sensitivity_Se": round(se, 2),
            "PositivePredictivity_P": round(pp, 2),
            "Specificity_Sp": round(sp, 2),
            "F1_Score": round(f1, 2),
        }

    return {
        "overall_accuracy_pct": round(overall_acc * 100.0, 2),
        "macro_f1_pct": round(macro_f1 * 100.0, 2),
        "weighted_f1_pct": round(weighted_f1 * 100.0, 2),
        "total_test_samples": total_samples,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }


def format_metrics_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Format per-class AAMI metrics into a clean pandas DataFrame.
    """
    rows = []
    for c, stats in metrics["per_class"].items():
        rows.append({
            "Class": c,
            "Support": stats["Support"],
            "TP": stats["TP"],
            "FP": stats["FP"],
            "FN": stats["FN"],
            "Sensitivity (Se %)": f"{stats['Sensitivity_Se']:.2f}%",
            "+Predictivity (+P %)": f"{stats['PositivePredictivity_P']:.2f}%",
            "Specificity (Sp %)": f"{stats['Specificity_Sp']:.2f}%",
            "F1-Score (%)": f"{stats['F1_Score']:.2f}%",
        })
    df = pd.DataFrame(rows)
    return df


def compare_models(
    models_dict: Dict[str, Dict[str, Any]],
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generate a side-by-side comparison table between multiple models (e.g. HMM vs RF vs GBDT).

    Parameters
    ----------
    models_dict : Dict[str, Dict[str, Any]]
        Mapping of model_name -> output dictionary from compute_aami_metrics.
    save_path : Optional[Path]
        If provided, writes table to CSV.

    Returns
    -------
    pd.DataFrame
        Comparative evaluation summary.
    """
    rows = []
    for model_name, res in models_dict.items():
        row = {
            "Model": model_name,
            "Overall Acc (%)": f"{res['overall_accuracy_pct']:.2f}%",
            "Macro F1 (%)": f"{res['macro_f1_pct']:.2f}%",
            "Weighted F1 (%)": f"{res['weighted_f1_pct']:.2f}%",
        }
        for c in AAMI_CLASSES:
            if c in res["per_class"]:
                row[f"{c} Se (%)"] = f"{res['per_class'][c]['Sensitivity_Se']:.1f}%"
                row[f"{c} +P (%)"] = f"{res['per_class'][c]['PositivePredictivity_P']:.1f}%"
                row[f"{c} F1 (%)"] = f"{res['per_class'][c]['F1_Score']:.1f}%"
        rows.append(row)

    df_comp = pd.DataFrame(rows)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_comp.to_csv(save_path, index=False)
        print(f"[Evaluation] Model comparison saved to results/{save_path.name}")

    return df_comp
