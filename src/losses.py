"""
losses.py — Hàm mất mát và điều chỉnh logit
============================================
Cung cấp các công cụ toán học numpy thuần cho:
  - softmax_np          : Tính softmax ổn định số học
  - compute_class_priors: Tính tần suất tiên nghiệm pi_k từ tập train
  - logit_adjusted_logits: Điều chỉnh logit theo Menon et al. (ICLR 2021)
  - smooth_labels       : Chuyển nhãn cứng thành soft labels (ε-Softmax / Label Smoothing)
  - ce_loss_np          : Cross-entropy loss numpy thuần (dùng trong optimizer tùy biến)
  - epsilon_softmax_loss_np: Hàm mất mát ε-Softmax (NeurIPS 2024)
"""

import numpy as np
from scipy.special import softmax as scipy_softmax


# ---------------------------------------------------------------------------
# Hàm cơ bản
# ---------------------------------------------------------------------------

def softmax_np(logits: np.ndarray) -> np.ndarray:
    """
    Softmax ổn định số học (tránh overflow) — tương đương scipy_softmax.

    Args:
        logits: (n_samples, n_classes) hoặc (n_classes,)

    Returns:
        Phân phối xác suất cùng shape với logits.
    """
    return scipy_softmax(logits, axis=-1)


def compute_class_priors(y: np.ndarray) -> np.ndarray:
    """
    Tính tần suất tiên nghiệm pi_k = count(k) / n từ vector nhãn.

    Args:
        y: Nhãn nguyên (int), shape (n_samples,).

    Returns:
        pi: Array shape (n_classes,) với pi[k] = tỷ lệ lớp k.

    Ví dụ:
        >>> pi = compute_class_priors(y_train)
        >>> # pi ~ [0.866, 0.112, 0.022] cho Bronze/Silver/Gold
    """
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    pi = np.zeros(n_classes)
    for cls, cnt in zip(classes, counts):
        pi[cls] = cnt
    pi = pi / pi.sum()
    return pi


# ---------------------------------------------------------------------------
# Logit Adjustment (ICLR 2021)
# ---------------------------------------------------------------------------

def logit_adjusted_logits(
    logits: np.ndarray,
    class_priors: np.ndarray,
    tau: float = 1.0,
) -> np.ndarray:
    """
    Điều chỉnh logit theo Menon et al. ICLR 2021 (inference-time variant):

        z_k^adj = z_k - tau * log(pi_k)

    Vì pi_k(Gold) << pi_k(Bronze), nên -tau*log(pi_Gold) > 0 → boosts Gold.
    Điều này mở rộng biên phân cách (Margin Shift) cho lớp thiểu số.

    Args:
        logits       : Raw logits (n_samples, n_classes) từ mô hình.
        class_priors : pi_k (n_classes,) tính từ tập train.
        tau          : Cường độ điều chỉnh; mặc định 1.0.

    Returns:
        adjusted_logits: (n_samples, n_classes)
    """
    # Tránh log(0) bằng cách clip nhỏ
    pi = np.clip(class_priors, 1e-8, 1.0)
    log_pi = np.log(pi)                    # shape (n_classes,)
    return logits - tau * log_pi           # broadcasting: (n, K) - (K,)


# ---------------------------------------------------------------------------
# ε-Softmax / Label Smoothing (NeurIPS 2024)
# ---------------------------------------------------------------------------

def smooth_labels(
    y: np.ndarray,
    n_classes: int = None,
    epsilon: float = 0.1,
    num_classes: int = None,
) -> np.ndarray:
    """
    Chuyển nhãn cứng (hard labels) thành soft labels theo ε-Softmax:

        y_smooth_k = (1 - ε) * I[y == k]  +  ε / K

    Trong đó K = n_classes. Thay vì mô hình học "chắc chắn" 100%,
    nó học phân phối mềm → tránh Overconfidence và kháng nhiễu nhãn.

    Args:
        y        : Nhãn nguyên (int), shape (n_samples,).
        n_classes: Số lớp K.
        n_classes: Số lớp K (hoặc num_classes).
        epsilon  : Độ làm mềm; mặc định 0.1.

    Returns:
        y_smooth: (n_samples, n_classes) — phân phối soft labels.
    """
    if n_classes is None:
        if num_classes is not None:
            n_classes = num_classes
        else:
            n_classes = len(np.unique(y))
    n = len(y)
    # One-hot encoding
    y_onehot = np.zeros((n, n_classes))
    y_onehot[np.arange(n), y] = 1.0
    # Label smoothing
    y_smooth = (1.0 - epsilon) * y_onehot + epsilon / n_classes
    return y_smooth


# ---------------------------------------------------------------------------
# Loss functions (numpy — dùng trong optimizer scipy của EpsilonSoftmaxClassifier)
# ---------------------------------------------------------------------------

def ce_loss_np(
    logits: np.ndarray,
    y: np.ndarray,
    l2_reg: float = 1e-4,
) -> float:
    """
    Standard Cross-Entropy Loss với L2 regularization.

    Dùng cho hàm mục tiêu truyền vào scipy.optimize.minimize.

    Args:
        logits : (n_samples, n_classes)
        y      : Hard labels (n_samples,) — int
        l2_reg : Hệ số regularization L2.

    Returns:
        loss scalar.
    """
    probs = softmax_np(logits)               # (n, K)
    n = len(y)
    log_p = np.log(probs[np.arange(n), y] + 1e-15)
    return -log_p.mean() + l2_reg * (logits ** 2).mean()


def epsilon_softmax_loss_np(
    logits: np.ndarray,
    y_smooth: np.ndarray,
    l2_reg: float = 1e-4,
) -> float:
    """
    ε-Softmax Loss: Cross-entropy với soft labels.

    loss = -sum_k [ y_smooth_k * log(p_k) ]  (trung bình trên samples)

    Args:
        logits  : (n_samples, n_classes)
        y_smooth: Soft labels (n_samples, n_classes) từ smooth_labels().
        l2_reg  : Hệ số L2.

    Returns:
        loss scalar.
    """
    probs = softmax_np(logits)                         # (n, K)
    log_p = np.log(probs + 1e-15)                     # (n, K)
    ce = -np.sum(y_smooth * log_p, axis=1).mean()     # scalar
    reg = l2_reg * (logits ** 2).mean()
    return ce + reg

