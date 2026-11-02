"""Classification metrics for TCMB downstream experiments."""
from typing import Sequence, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

def classification_metrics(y_true: Sequence[Any], y_pred: Sequence[Any], labels=None):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty prediction set.")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "n_examples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()
            if labels is not None else confusion_matrix(y_true, y_pred).tolist(),
    }
