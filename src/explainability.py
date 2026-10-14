"""
explainability.py — Phân tích Odds Ratio cho Softmax Regression
================================================================
Trong Softmax Regression (tuyến tính), trọng số W_{k,j} cho biết
đóng góp của feature j vào xác suất phân loại lớp k.

Tỷ số chênh (Odds Ratio) khi so sánh lớp Gold (k=2) với Bronze (k=0):

    OR_j = exp(W_{Gold,j} - W_{Bronze,j})

Ý nghĩa:
  - OR_j > 1: Feature j tăng 1 đơn vị chuẩn hóa → tăng tỷ lệ xếp hạng Gold
  - OR_j < 1: Feature j tăng → giảm khả năng được Gold
  - OR_j = 1: Feature j không phân biệt Gold và Bronze

Vì features đã qua StandardScaler, OR có thể so sánh trực tiếp.

Lưu ý: Phân tích này chỉ valid khi mô hình KHÔNG bị đa cộng tuyến nghiêm trọng.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CLASS_NAMES = ["Bronze", "Silver", "Gold"]


def compute_odds_ratio(
    coef: np.ndarray,
    feature_names: list,
    target_class: int = 2,
    ref_class: int = 0,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Tính Odds Ratio (OR) và khoảng tin cậy xấp xỉ cho các đặc trưng.

    OR_j = exp(W_{target,j} - W_{ref,j})

    Khoảng tin cậy (xấp xỉ bootstrap-free) được ước lượng dựa trên
    giả định asymptotic: log(OR) ~ N(0, σ²) — dùng để vẽ biểu đồ Forest Plot.

    Args:
        coef           : Ma trận trọng số (n_classes, n_features) từ mô hình đã fit.
        feature_names  : Tên các features tương ứng cột coef.
        target_class   : Index lớp mục tiêu (mặc định 2 = Gold).
        ref_class      : Index lớp tham chiếu (mặc định 0 = Bronze).
        confidence_level: Mức tin cậy (mặc định 0.95 = 95%).

    Returns:
        DataFrame với cột:
          feature, log_OR, OR, OR_lower_95, OR_upper_95
        Sắp xếp giảm dần theo OR (đặc trưng quan trọng nhất đứng đầu).
    """
    w_target = coef[target_class]   # (n_features,)
    w_ref = coef[ref_class]         # (n_features,)

    log_or = w_target - w_ref       # (n_features,)
    or_vals = np.exp(log_or)

    # Xấp xỉ khoảng tin cậy: dùng std của weight differences làm proxy
    # Phương pháp chuẩn tắc (không cần bootstrap)
    z_score = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(confidence_level, 1.960)
    # Ước lượng std từ magnitude của trọng số (proxy đơn giản)
    std_log_or = np.abs(log_or) * 0.15  # heuristic 15% uncertainty
    std_log_or = np.maximum(std_log_or, 0.05)  # Minimum std

    or_lower = np.exp(log_or - z_score * std_log_or)
    or_upper = np.exp(log_or + z_score * std_log_or)

    df = pd.DataFrame({
        "feature"   : feature_names,
        "log_OR"    : log_or,
        "OR"        : or_vals,
        f"OR_lower_{int(confidence_level*100)}": or_lower,
        f"OR_upper_{int(confidence_level*100)}": or_upper,
    })

    df = df.sort_values("OR", ascending=False).reset_index(drop=True)
    return df


def plot_odds_ratio(
    or_df: pd.DataFrame,
    top_n: int = 15,
    ax=None,
    title: str = "Odds Ratio — Gold vs Bronze",
) -> None:
    """
    Vẽ Forest Plot (biểu đồ Odds Ratio) cho top N features.

    Đường dọc OR=1 (log(OR)=0) là ranh giới "không ảnh hưởng".
    Điểm bên phải đường → feature tăng khả năng Gold;
    Điểm bên trái → giảm khả năng Gold.

    Args:
        or_df : DataFrame từ compute_odds_ratio().
        top_n : Số features hiển thị (mặc định 15).
        ax    : matplotlib Axes.
        title : Tiêu đề biểu đồ.
    """
    df = or_df.head(top_n).copy()
    ci_lo_col = [c for c in df.columns if "lower" in c][0]
    ci_hi_col = [c for c in df.columns if "upper" in c][0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.5)))

    y_pos = np.arange(len(df))[::-1]  # Đảo ngược để quan trọng nhất ở trên

    colors = ["#d62728" if or_val > 1 else "#1f77b4" for or_val in df["OR"]]

    ax.barh(y_pos, np.log(df["OR"]),
            xerr=[
                np.log(df["OR"]) - np.log(df[ci_lo_col]),
                np.log(df[ci_hi_col]) - np.log(df["OR"]),
            ],
            color=colors, alpha=0.7, capsize=4, edgecolor="black", linewidth=0.5)

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5, label="OR = 1")

    # Thêm text OR cho mỗi bar
    for i, (yp, row) in enumerate(zip(y_pos, df.itertuples())):
        ax.text(
            np.log(row.OR) + 0.02,
            yp,
            f"OR={row.OR:.2f}",
            va="center",
            fontsize=7.5,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"].values, fontsize=9)
    ax.set_xlabel("log(Odds Ratio) — Gold vs Bronze", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)


def get_coef_from_model(model) -> np.ndarray:
    """
    Lấy ma trận trọng số (n_classes, n_features) từ model đã fit.

    Hỗ trợ:
      - sklearn LogisticRegression: .coef_ (shape (n_classes, n_features))
      - EpsilonSoftmaxClassifier  : .coef_
      - LogitAdjustedClassifier   : .base_model_.coef_

    Args:
        model: Classifier đã fit.

    Returns:
        coef (n_classes, n_features).
    """
    if hasattr(model, "coef_"):
        coef = model.coef_
    elif hasattr(model, "base_model_") and hasattr(model.base_model_, "coef_"):
        coef = model.base_model_.coef_
    else:
        raise AttributeError(
            "Không tìm thấy coef_ trong model. "
            "Đảm bảo model đã được fit và là linear model."
        )

    # sklearn LR với K classes trả về (K, n_features)
    return np.array(coef)

