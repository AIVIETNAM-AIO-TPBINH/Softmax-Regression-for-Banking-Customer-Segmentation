"""
models.py — Classifier sklearn-compatible với loss tùy biến
============================================================
Cung cấp 2 mô hình tùy biến bổ sung vào bộ 7 cấu hình ablation:

  1. LogitAdjustedClassifier
     Wrapper quanh sklearn LogisticRegression.
     Áp dụng Logit Adjustment (ICLR 2021) tại thời điểm inference:
         z_k^adj = z_k - tau * log(pi_k)
     Phương pháp này equivalent với training-time LA cho mô hình
     linear được huấn luyện đúng hàm mục tiêu (Menon et al., 2021).

  2. EpsilonSoftmaxClassifier
     Softmax Regression tùy biến với Label Smoothing (ε-Softmax).
     Sử dụng scipy.optimize.minimize (L-BFGS-B) để tối ưu:
         loss = -sum_k [y_smooth_k * log(p_k)] + λ||W||²
     Đây là sklearn-compatible estimator (fit/predict/predict_proba).

Cả hai đều tuân theo scikit-learn API để tích hợp dễ dàng.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax as scipy_softmax, logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from src.losses import (
    compute_class_priors,
    logit_adjusted_logits,
    smooth_labels,
)


# ---------------------------------------------------------------------------
# 1. Logit Adjusted Classifier
# ---------------------------------------------------------------------------

class LogitAdjustedClassifier(BaseEstimator, ClassifierMixin):
    """
    Logit Adjustment Classifier (Menon et al., ICLR 2021).

    Huấn luyện: sklearn LogisticRegression (multinomial, L2, solver=lbfgs).
    Inference : z_k^adj = z_k - tau * log(pi_k)

    Parameters:
        tau    : Cường độ Margin Shift; tau=1.0 (paper default).
        C      : Hệ số inverse regularization L2 (sklearn convention).
        max_iter: Số vòng lặp tối đa tối ưu.
        random_state: Seed.
    """

    def __init__(
        self,
        tau: float = 1.0,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.tau = tau
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Huấn luyện mô hình gốc và tính class priors từ tập train.
        """
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.class_priors_ = compute_class_priors(y)

        self.base_model_ = LogisticRegression(
            # # multi_class='multinomial', (auto in modern sklearn) (auto in modern sklearn)
            solver="lbfgs",
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.base_model_.fit(X, y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Trả về logit đã được điều chỉnh Logit Adjustment."""
        check_is_fitted(self)
        raw_logits = self.base_model_.decision_function(X)  # (n, K)
        # sklearn multinomial trả về (n, K) khi K > 2
        if raw_logits.ndim == 1:
            # Binary fallback (không xảy ra với K=3, nhưng để an toàn)
            raw_logits = np.column_stack([-raw_logits / 2, raw_logits / 2])
        return logit_adjusted_logits(raw_logits, self.class_priors_, tau=self.tau)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Xác suất sau Logit Adjustment."""
        adj_logits = self.decision_function(X)
        return scipy_softmax(adj_logits, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Nhãn dự đoán = argmax logit đã điều chỉnh."""
        return self.classes_[self.decision_function(X).argmax(axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return (self.predict(X) == y).mean()


# ---------------------------------------------------------------------------
# 2. Epsilon Softmax Classifier (Label Smoothing via scipy)
# ---------------------------------------------------------------------------

class EpsilonSoftmaxClassifier(BaseEstimator, ClassifierMixin):
    """
    Softmax Regression với ε-Softmax Loss (Label Smoothing).

    Tối ưu hàm mất mát:
        L = -sum_k [y_smooth_k * log(p_k)] + (1/C) * ||W||²
    với y_smooth = (1-ε)*y_onehot + ε/K

    Sử dụng scipy.optimize.minimize (L-BFGS-B), tương đương với
    sklearn's LogisticRegression solver nhưng cho phép custom loss.

    Parameters:
        epsilon     : Hệ số làm mềm nhãn; 0 = standard CE.
        C           : Inverse L2 regularization (như sklearn).
        max_iter    : Số vòng lặp L-BFGS-B tối đa.
        tol         : Ngưỡng hội tụ.
        random_state: Seed khởi tạo trọng số.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        C: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        self.epsilon = epsilon
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    # ---- Hàm mất mát + gradient (dạng vectorized) ----

    def _loss_grad(
        self,
        W_flat: np.ndarray,
        X: np.ndarray,
        y_smooth: np.ndarray,
    ):
        """
        Tính loss và gradient cho scipy.optimize.minimize.

        W_flat : (n_classes * n_features,) — trọng số được flatten
        X      : (n_samples, n_features)
        y_smooth: (n_samples, n_classes) — soft labels
        """
        n_samples, n_features = X.shape
        n_classes = y_smooth.shape[1]
        l2_reg = 1.0 / (2.0 * self.C * n_samples)

        W = W_flat.reshape(n_classes, n_features)   # (K, d)
        logits = X @ W.T                             # (n, K)

        # Ổn định số học: trừ max theo hàng
        log_sum_exp = logsumexp(logits, axis=1, keepdims=True)  # (n, 1)
        log_probs = logits - log_sum_exp                         # (n, K)
        probs = np.exp(log_probs)                                # (n, K)

        # Loss: CE với soft labels + L2
        ce = -np.sum(y_smooth * log_probs) / n_samples
        reg = l2_reg * np.sum(W ** 2)
        loss = ce + reg

        # Gradient: (probs - y_smooth)^T @ X / n
        diff = (probs - y_smooth) / n_samples        # (n, K)
        grad_W = diff.T @ X + 2.0 * l2_reg * W      # (K, d)
        grad_flat = grad_W.ravel()

        return loss, grad_flat

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Huấn luyện bằng scipy L-BFGS-B với ε-Softmax loss."""
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        n_features = X.shape[1]

        # Soft labels
        y_smooth = smooth_labels(y, self.n_classes_, epsilon=self.epsilon)

        # Khởi tạo trọng số
        rng = np.random.RandomState(self.random_state)
        W0 = rng.randn(self.n_classes_ * n_features) * 0.01

        result = minimize(
            fun=self._loss_grad,
            x0=W0,
            args=(X, y_smooth),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
        )

        self.coef_ = result.x.reshape(self.n_classes_, n_features)
        self.converged_ = result.success
        self.n_iter_ = result.nit

        if not self.converged_:
            import warnings
            warnings.warn(
                f"EpsilonSoftmaxClassifier chưa hội tụ sau {self.max_iter} vòng lặp. "
                "Thử tăng max_iter.",
                RuntimeWarning,
            )
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Raw logits (n, K)."""
        check_is_fitted(self, "coef_")
        return X @ self.coef_.T

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Xác suất softmax (n, K)."""
        return scipy_softmax(self.decision_function(X), axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Nhãn dự đoán."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return (self.predict(X) == y).mean()

