"""
calibration.py — Temperature Scaling cho hiệu chuẩn xác suất
=============================================================
Temperature Scaling là phương pháp hiệu chuẩn hậu xử lý đơn giản nhất
(Guo et al., ICML 2017): chia toàn bộ logit cho một hằng số T > 0:

    p_k = softmax(z / T)

Với T > 1: làm mềm phân phối (giảm overconfidence).
Với T < 1: làm sắc phân phối (hiếm gặp).

T* được tối ưu trên tập Validation bằng thuật toán L-BFGS-B
bằng cách tối thiểu hóa NLL (Negative Log-Likelihood):

    T* = argmin_T  -1/n * sum_i log p_{y_i}(z_i / T)

Ưu điểm: Không thay đổi thứ tự xếp hạng → accuracy 100% được bảo toàn.
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.special import softmax as scipy_softmax


class TemperatureScaler:
    """
    Temperature Scaling Calibrator.

    Sử dụng:
        scaler = TemperatureScaler()
        scaler.fit(logits_val, y_val)   # Tìm T* trên validation set
        proba_cal = scaler.predict_proba(logits_test)  # Dự đoán đã hiệu chuẩn

    Attributes:
        T_     : Nhiệt độ tối ưu T* (scalar > 0) sau khi fit().
        nll_   : NLL tối thiểu đạt được trên validation set.
    """

    def __init__(self):
        self.T_ = 1.0    # Mặc định T=1 → không thay đổi
        self.nll_ = None

    # ---- NLL objective ----

    @staticmethod
    def _nll(T: float, logits: np.ndarray, y: np.ndarray) -> float:
        """
        Negative Log-Likelihood với temperature T.

        Args:
            T      : Scalar nhiệt độ (> 0).
            logits : (n_samples, n_classes) — raw logits từ mô hình đã đóng băng.
            y      : Nhãn thực (int), shape (n_samples,).

        Returns:
            NLL scalar.
        """
        T = max(T, 1e-6)  # Tránh chia cho 0
        scaled = logits / T
        log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        nll = -log_probs[np.arange(len(y)), y].mean()
        return float(nll)

    # ---- Fit ----

    def fit(
        self,
        logits_val: np.ndarray,
        y_val: np.ndarray,
        method: str = "bounded",
    ):
        """
        Tìm T* = argmin_T NLL(logits_val / T, y_val) trên validation set.

        Sử dụng scipy.optimize.minimize_scalar (golden section / bounded)
        — 1D optimization, rất nhanh.

        Args:
            logits_val: Raw logits trên validation set (n, K).
            y_val     : Nhãn thực validation set.
            method    : "bounded" (mặc định, tìm trong [0.05, 10.0])
                        hoặc "brent" (không giới hạn).

        Returns:
            self (để chain: scaler.fit(...).predict_proba(...))
        """
        result = minimize_scalar(
            fun=lambda T: self._nll(T, logits_val, y_val),
            bounds=(0.05, 10.0),
            method=method,
        )
        self.T_ = float(result.x)
        self.nll_ = float(result.fun)
        print(f"[TemperatureScaler] T* = {self.T_:.4f} | NLL = {self.nll_:.6f}")
        return self

    # ---- Transform ----

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Chia logits cho T* → scaled logits."""
        return logits / max(self.T_, 1e-6)

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Softmax(logits / T*) — xác suất đã hiệu chuẩn."""
        return scipy_softmax(self.transform(logits), axis=1)

    def predict(self, logits: np.ndarray) -> np.ndarray:
        """Nhãn dự đoán (argmax — không thay đổi so với trước calibration)."""
        return self.predict_proba(logits).argmax(axis=1)

    # ---- Utility ----

    def summary(self) -> dict:
        """Trả về thông tin tóm tắt."""
        return {"T_optimal": self.T_, "nll_val": self.nll_}


def get_logits_from_sklearn(model, X: np.ndarray) -> np.ndarray:
    """
    Lấy raw logits từ sklearn estimator.

    Với sklearn LogisticRegression: dùng decision_function().
    Với custom model có decision_function(): dùng trực tiếp.
    Nếu không có: log(predict_proba()) làm logits gần đúng.

    Args:
        model: Đã fit sklearn estimator.
        X    : Feature matrix.

    Returns:
        logits (n, K).
    """
    if hasattr(model, "decision_function"):
        logits = model.decision_function(X)
        # sklearn trả về (n,) cho binary; (n, K) cho K >= 3
        if logits.ndim == 1:
            logits = np.column_stack([-logits / 2, logits / 2])
        return logits
    else:
        # Fallback: dùng log của predict_proba
        proba = model.predict_proba(X)
        return np.log(proba + 1e-15)

