"""
utils.py — Hàm tiện ích dùng chung cho toàn dự án
===================================================
  - set_seed: Cố định random seed toàn cục
  - save_results / load_results: Lưu/tải kết quả JSON
  - plot_reliability_diagram: Biểu đồ hiệu chuẩn xác suất
  - plot_coverage_accuracy: Đường Coverage–Accuracy
  - plot_confusion_matrix: Ma trận nhầm lẫn
"""

import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path


# ---------------------------------------------------------------------------
# Seed Management
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Cố định random seed cho numpy và random để đảm bảo tái lập thí nghiệm."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def save_results(results: dict, path: str) -> None:
    """Lưu dict kết quả thực nghiệm ra file JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else (o.tolist() if hasattr(o, 'tolist') else str(o)))
    print(f"[INFO] Đã lưu kết quả vào: {path}")


def load_results(path: str) -> dict:
    """Tải kết quả thực nghiệm từ file JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot: Reliability Diagram (Calibration)
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    ax=None,
    title: str = "Reliability Diagram",
    class_names: list = None,
) -> None:
    """
    Vẽ Reliability Diagram (biểu đồ đường cong hiệu chuẩn) cho bài toán đa lớp.

    Với mỗi lớp k, vẽ đường confidence vs. accuracy trung bình trong từng bin.
    Đường diagonal hoàn hảo tương ứng với mô hình được hiệu chuẩn tốt.

    Args:
        y_true  : Nhãn thực (int, shape [n_samples]).
        y_proba : Xác suất dự đoán (shape [n_samples, n_classes]).
        n_bins  : Số lượng bins chia đều khoảng [0, 1].
        ax      : matplotlib Axes; nếu None sẽ tạo figure mới.
        title   : Tiêu đề biểu đồ.
        class_names: Tên từng lớp (Bronze, Silver, Gold); nếu None dùng 0,1,2.
    """
    n_classes = y_proba.shape[1]
    if class_names is None:
        class_names = [str(k) for k in range(n_classes)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for k in range(n_classes):
        prob_k = y_proba[:, k]
        is_k = (y_true == k).astype(float)

        bin_accs, bin_confs = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (prob_k >= lo) & (prob_k < hi)
            if mask.sum() > 0:
                bin_accs.append(is_k[mask].mean())
                bin_confs.append(prob_k[mask].mean())

        color = colors[k % len(colors)]
        ax.plot(bin_confs, bin_accs, "o-", label=class_names[k],
                color=color, linewidth=2, markersize=5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly Calibrated")
    ax.set_xlabel("Confidence trung bình trong bin", fontsize=11)
    ax.set_ylabel("Accuracy thực tế trong bin", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Plot: Coverage–Accuracy Curve
# ---------------------------------------------------------------------------

def plot_coverage_accuracy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray = None,
    ax=None,
    title: str = "Coverage–Accuracy Curve",
) -> tuple:
    """
    Vẽ đường Coverage–Accuracy: với mỗi ngưỡng độ tin cậy (confidence threshold),
    tính % khách hàng được tự động phân hạng (Coverage) và accuracy trên nhóm đó.

    Điểm vận hành lý tưởng: Coverage cao + Accuracy cao.

    Args:
        y_true     : Nhãn thực (int, shape [n_samples]).
        y_proba    : Xác suất dự đoán (shape [n_samples, n_classes]).
        thresholds : Mảng ngưỡng cần khảo sát; mặc định np.linspace(0.5, 0.99, 50).
        ax         : matplotlib Axes.
        title      : Tiêu đề biểu đồ.

    Returns:
        (thresholds, coverages, accuracies) để tiện phân tích thêm.
    """
    if thresholds is None:
        thresholds = np.linspace(0.50, 0.99, 50)

    max_prob = y_proba.max(axis=1)
    y_pred_all = y_proba.argmax(axis=1)
    n = len(y_true)

    coverages, accuracies = [], []
    for thr in thresholds:
        mask = max_prob >= thr
        coverage = mask.sum() / n
        if mask.sum() > 0:
            acc = (y_pred_all[mask] == y_true[mask]).mean()
        else:
            acc = np.nan
        coverages.append(coverage)
        accuracies.append(acc)

    coverages = np.array(coverages)
    accuracies = np.array(accuracies)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(coverages * 100, accuracies * 100, "b-o", linewidth=2,
            markersize=4, label="Coverage–Accuracy")
    ax.axhline(85, color="red", linestyle="--", linewidth=1, label="Ngưỡng Accuracy 85%")
    ax.set_xlabel("Coverage (% khách hàng tự động phân hạng)", fontsize=11)
    ax.set_ylabel("Accuracy trên nhóm được tự động phân hạng (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(50, 102)

    return thresholds, coverages, accuracies


# ---------------------------------------------------------------------------
# Plot: Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list,
    ax=None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """
    Vẽ Confusion Matrix đẹp dưới dạng heatmap.

    Args:
        cm      : Ma trận nhầm lẫn (shape [n_classes, n_classes]).
        classes : Danh sách tên lớp.
        ax      : matplotlib Axes.
        title   : Tiêu đề.
        normalize: Nếu True, chuẩn hóa theo hàng (tỷ lệ Recall).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    sns.heatmap(
        cm_display, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel("Dự đoán", fontsize=11)
    ax.set_ylabel("Thực tế", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

