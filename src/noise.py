"""
noise.py — Giả lập nhiễu nhãn liền kề (Adjacent Label Noise)
=============================================================
Chỉ lật nhãn giữa các bậc kề nhau (Bronze↔Silver, Silver↔Gold).
Tránh lật trực tiếp Bronze↔Gold để không xóa sổ hoàn toàn lớp Gold.

Ứng dụng: Kiểm định độ bền vững mô hình dưới nhiễu nghiệp vụ thực tế
(nhân viên mã hóa sai hạng liền kề khi nhập liệu hồ sơ khách hàng).
"""

import numpy as np
from typing import Tuple


# Ánh xạ bậc hạng: 0=Bronze, 1=Silver, 2=Gold
CLASS_NAMES = {0: "Bronze", 1: "Silver", 2: "Gold"}

# Các cặp (from, to) cho phép lật nhãn liền kề
ADJACENT_PAIRS = [
    (0, 1),  # Bronze -> Silver
    (1, 0),  # Silver -> Bronze
    (1, 2),  # Silver -> Gold
    (2, 1),  # Gold   -> Silver
]


def inject_adjacent_noise(
    y: np.ndarray,
    noise_rate: float = 0.0,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Chèn nhiễu nhãn liền kề vào vector nhãn y.

    Với mỗi mẫu có xác suất noise_rate, nhãn bị lật sang bậc kề nhau
    theo phân phối đều giữa các cặp liền kề hợp lệ.

    Ràng buộc:
      - Lớp Bronze (0): chỉ có thể bị lật thành Silver (1).
      - Lớp Silver (1): có thể bị lật thành Bronze (0) hoặc Gold (2).
      - Lớp Gold   (2): chỉ có thể bị lật thành Silver (1).

    Args:
        y          : Nhãn nguyên gốc (int), shape (n_samples,).
        noise_rate : Tỷ lệ nhiễu ∈ [0.0, 1.0]; ví dụ 0.10 = 10%.
        seed       : Random seed để tái lập.
        verbose    : In thống kê trước/sau khi chèn nhiễu.

    Returns:
        y_noisy: Nhãn sau khi chèn nhiễu.
        info   : Dict chứa thống kê số lượng nhãn bị lật.

    Ví dụ:
        >>> y_noisy, info = inject_adjacent_noise(y_train, noise_rate=0.10, seed=42)
        >>> print(info["n_flipped"])  # ~10% của len(y_train)
    """
    assert 0.0 <= noise_rate <= 1.0, "noise_rate phải trong khoảng [0, 1]."
    rng = np.random.RandomState(seed)
    y_noisy = y.copy()

    if noise_rate == 0.0:
        info = {"noise_rate": 0.0, "n_flipped": 0, "flip_counts": {}}
        return y_noisy, info

    n = len(y)
    flip_mask = rng.rand(n) < noise_rate   # True = mẫu này sẽ bị lật
    flip_counts = {str(pair): 0 for pair in ADJACENT_PAIRS}

    for i in np.where(flip_mask)[0]:
        orig = y[i]
        # Lấy danh sách lớp kề hợp lệ
        neighbors = [to for (frm, to) in ADJACENT_PAIRS if frm == orig]
        if len(neighbors) == 0:
            continue
        new_label = rng.choice(neighbors)
        y_noisy[i] = new_label
        flip_counts[str((int(orig), int(new_label)))] += 1

    info = {
        "noise_rate"  : noise_rate,
        "n_flipped"   : int(flip_mask.sum()),
        "flip_counts" : flip_counts,
    }

    if verbose:
        print(f"\n[Noise Injection] noise_rate={noise_rate:.0%}, "
              f"n_flipped={info['n_flipped']} / {n}")
        print("  Phân bố nhãn GỐC :", {CLASS_NAMES[k]: int((y == k).sum())
                                        for k in [0, 1, 2]})
        print("  Phân bố nhãn NHIỄU:", {CLASS_NAMES[k]: int((y_noisy == k).sum())
                                         for k in [0, 1, 2]})
        print("  Chi tiết lật nhãn  :", flip_counts)

    return y_noisy, info


def noise_schedule(
    y: np.ndarray,
    noise_rates: list = None,
    seed: int = 42,
) -> dict:
    """
    Sinh nhiễu cho nhiều mức tỷ lệ noise_rate khác nhau.

    Args:
        y          : Nhãn gốc.
        noise_rates: Danh sách các mức cần thử; mặc định [0, 0.10, 0.20, 0.30].
        seed       : Random seed gốc (mỗi mức dùng seed + idx).

    Returns:
        Dict {noise_rate: (y_noisy, info)} cho mỗi mức noise.
    """
    if noise_rates is None:
        noise_rates = [0.0, 0.10, 0.20, 0.30]

    results = {}
    for idx, rate in enumerate(noise_rates):
        y_noisy, info = inject_adjacent_noise(
            y, noise_rate=rate, seed=seed + idx, verbose=False
        )
        results[rate] = (y_noisy, info)
        print(f"  noise_rate={rate:.0%}: {info['n_flipped']} nhãn bị lật")

    return results

