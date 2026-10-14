"""
metrics.py — Bộ đo lường đầy đủ cho bài toán phân hạng đa lớp mất cân bằng
=============================================================================
Các hàm đo lường:
  - balanced_accuracy     : Accuracy cân bằng theo lớp (không bị Bronze áp đảo)
  - macro_f1              : F1-score trung bình macro
  - per_class_recall      : Recall từng lớp (Bronze, Silver, Gold)
  - gold_pr_auc           : PR-AUC riêng lớp VIP Gold
  - classwise_ece         : ECE từng lớp (Expected Calibration Error)
  - overall_ece           : ECE tổng thể (One-vs-Rest)
  - extreme_twostep_rate  : Tỷ lệ lỗi "nhảy 2 bậc" (Gold ↔ Bronze)
  - compute_full_report   : Tổng hợp toàn bộ metrics vào 1 dict
"""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
)

CLASS_NAMES = ["Bronze", "Silver", "Gold"]
N_CLASSES = 3


# ---------------------------------------------------------------------------
# Metrics phân loại
# ---------------------------------------------------------------------------

def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Balanced Accuracy = trung bình Recall của từng lớp.
    Không bị lệch bởi lớp Bronze chiếm 86.6%.
    """
    return float(balanced_accuracy_score(y_true, y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 (trọng số đều giữa 3 lớp)."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
) -> dict:
    """
    Recall từng lớp: TPR_k = TP_k / (TP_k + FN_k).

    Returns:
        Dict {"Bronze": r0, "Silver": r1, "Gold": r2}
    """
    if class_names is None:
        class_names = CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    recalls = {}
    for k, name in enumerate(class_names):
        denom = cm[k, :].sum()
        recalls[name] = float(cm[k, k] / denom) if denom > 0 else 0.0
    return recalls


def gold_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    PR-AUC (Precision–Recall Area Under Curve) riêng cho lớp Gold (k=2).

    PR-AUC đặc biệt hữu ích cho lớp thiểu số vì không bị ROC-AUC che giấu
    bởi Negative lớn (Bronze).

    Args:
        y_true  : Nhãn thực (int).
        y_proba : (n_samples, n_classes) — xác suất dự đoán.

    Returns:
        pr_auc scalar ∈ [0, 1].
    """
    y_binary = (y_true == 2).astype(int)  # 1 nếu là Gold, 0 nếu không
    gold_proba = y_proba[:, 2]
    precision, recall, _ = precision_recall_curve(y_binary, gold_proba)
    return float(auc(recall, precision))


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def classwise_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    class_names: list = None,
) -> dict:
    """
    Classwise ECE: Tính ECE độc lập cho từng lớp (One-vs-Rest).

    Tránh bị lớp Bronze (86.6%) áp đảo chỉ số ECE tổng thể.

    ECE_k = sum_b [ (n_b/n) * |acc_b - conf_b| ]

    Args:
        y_true  : Nhãn thực (int).
        y_proba : (n_samples, n_classes).
        n_bins  : Số bins chia đều [0, 1].

    Returns:
        Dict {"Bronze": ece0, "Silver": ece1, "Gold": ece2, "mean": ece_mean}
    """
    if class_names is None:
        class_names = CLASS_NAMES

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    eces = {}

    for k, name in enumerate(class_names):
        prob_k = y_proba[:, k]
        is_k = (y_true == k).astype(float)
        n = len(y_true)
        ece = 0.0

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (prob_k >= lo) & (prob_k < hi)
            if mask.sum() == 0:
                continue
            conf_b = prob_k[mask].mean()
            acc_b = is_k[mask].mean()
            ece += (mask.sum() / n) * abs(acc_b - conf_b)

        eces[name] = float(ece)

    eces["mean"] = float(np.mean(list(eces.values())))
    return eces


def overall_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    ECE tổng thể: dùng xác suất của lớp được dự đoán (confidence = max prob).

    Args:
        y_true  : Nhãn thực.
        y_proba : (n_samples, n_classes).

    Returns:
        ECE scalar.
    """
    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc_b = (predictions[mask] == y_true[mask]).mean()
        conf_b = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc_b - conf_b)

    return float(ece)


# ---------------------------------------------------------------------------
# Lỗi nhảy 2 bậc (Extreme Two-Step Error)
# ---------------------------------------------------------------------------

def extreme_twostep_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Tỷ lệ lỗi "nhảy 2 bậc cực đoan": nhầm lẫn giữa Gold (2) và Bronze (0).

    Đây là loại lỗi nghiêm trọng nhất — ví dụ phân loại khách VIP thành
    Bronze thông thường hoặc ngược lại.

    Returns:
        Tỷ lệ lỗi extreme (scalar ∈ [0, 1]).
    """
    n = len(y_true)
    # Lỗi: (thực Gold → dự đoán Bronze) hoặc (thực Bronze → dự đoán Gold)
    extreme_errors = (
        ((y_true == 2) & (y_pred == 0)) |
        ((y_true == 0) & (y_pred == 2))
    )
    return float(extreme_errors.sum() / n)


# ---------------------------------------------------------------------------
# Tổng hợp
# ---------------------------------------------------------------------------

def compute_full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Tổng hợp toàn bộ metrics vào một dict duy nhất.

    Returns:
        Dict với các key:
          balanced_accuracy, macro_f1,
          recall_bronze, recall_silver, recall_gold,
          gold_pr_auc,
          ece_bronze, ece_silver, ece_gold, ece_mean,
          overall_ece, extreme_twostep_rate
    """
    recall = per_class_recall(y_true, y_pred)
    cw_ece = classwise_ece(y_true, y_proba, n_bins=n_bins)

    report = {
        "balanced_accuracy"  : balanced_accuracy(y_true, y_pred),
        "macro_f1"           : macro_f1(y_true, y_pred),
        "recall_bronze"      : recall["Bronze"],
        "recall_silver"      : recall["Silver"],
        "recall_gold"        : recall["Gold"],
        "gold_pr_auc"        : gold_pr_auc(y_true, y_proba),
        "ece_bronze"         : cw_ece["Bronze"],
        "ece_silver"         : cw_ece["Silver"],
        "ece_gold"           : cw_ece["Gold"],
        "ece_mean"           : cw_ece["mean"],
        "overall_ece"        : overall_ece(y_true, y_proba, n_bins=n_bins),
        "extreme_twostep_rate": extreme_twostep_rate(y_true, y_pred),
    }
    return report


def print_report(report: dict, model_name: str = "Model") -> None:
    """In report đẹp ra console."""
    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  Balanced Accuracy  : {report['balanced_accuracy']:.4f}")
    print(f"  Macro F1           : {report['macro_f1']:.4f}")
    print(f"  Recall Bronze      : {report['recall_bronze']:.4f}")
    print(f"  Recall Silver      : {report['recall_silver']:.4f}")
    print(f"  Recall Gold  ★     : {report['recall_gold']:.4f}")
    print(f"  Gold PR-AUC        : {report['gold_pr_auc']:.4f}")
    print(f"  ECE (mean CW)      : {report['ece_mean']:.4f}")
    print(f"  ECE overall        : {report['overall_ece']:.4f}")
    print(f"  Extreme 2-step Err : {report['extreme_twostep_rate']:.4f}")
    print(f"{'='*55}\n")

