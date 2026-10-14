"""
preprocessing.py — Pipeline tiền xử lý dữ liệu ngân hàng
==========================================================
Bao gồm:
  - load_raw_data       : Tải file Excel 80k dòng
  - feature_engineering : Tạo đặc trưng tài chính phái sinh
  - drop_leakage_cols   : Xóa tenure_ye, engagement_score và định danh
  - encode_categoricals : One-hot / ordinal encoding cho biến phân loại
  - vif_check           : Kiểm tra Variance Inflation Factor (VIF < 5)
  - split_stratified    : Chia Stratified Train/Val/Test (70/15/15)
  - run_full_pipeline   : Chạy toàn bộ pipeline và lưu processed data

Sau pipeline, biến mục tiêu là loyalty_level được encode:
    Bronze = 0 | Silver = 1 | Gold = 2
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------

LABEL_COL = "loyalty_level"
LABEL_MAP = {"Bronze": 0, "Silver": 1, "Gold": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

# Các cột định danh + rò rỉ chính sách — loại bỏ bắt buộc
LEAKAGE_COLS = ["tenure_ye", "engagement_score"]
IDENTITY_COLS = ["id", "full_name", "address", "customer_id", "name", "last_active_date", "created_date"]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Tải file Excel dữ liệu ngân hàng.

    Args:
        path: Đường dẫn tuyệt đối đến file .xlsx.

    Returns:
        DataFrame gốc (80,000 dòng × 26 cột).
    """
    path = Path(path)
    print(f"[INFO] Đang tải dữ liệu từ: {path.name}")
    df = pd.read_excel(path, engine="openpyxl")
    print(f"[INFO] Shape: {df.shape} | Cột: {list(df.columns)}")
    print(f"[INFO] Phân phối nhãn:\n{df[LABEL_COL].value_counts(normalize=True).round(4)}")
    return df


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các đặc trưng tài chính phái sinh có ý nghĩa nghiệp vụ.

    Các đặc trưng mới:
      - balance_to_income  : Tỷ lệ số dư / thu nhập năm (đo tích lũy)
      - service_per_card   : Số sản phẩm / thẻ tín dụng (đo đa dạng hóa)
      - log_balance        : Log1p(balance) — giảm skewness số dư
      - log_monthly_ir     : Log1p(monthly_ir) — giảm skewness thu nhập

    Args:
        df: DataFrame có cột balance, monthly_ir, num_of_products, credit_card.

    Returns:
        DataFrame đã bổ sung các cột mới (inplace copy).
    """
    df = df.copy()

    # Tỷ lệ số dư / thu nhập hàng năm
    if "balance" in df.columns and "monthly_ir" in df.columns:
        annual_income = df["monthly_ir"] * 12
        df["balance_to_income"] = df["balance"] / annual_income.replace(0, np.nan)
        df["balance_to_income"] = df["balance_to_income"].fillna(0.0)

    # Số sản phẩm / số thẻ — dùng tên cột thực tế: nums_service, nums_card
    prod_col = "nums_service" if "nums_service" in df.columns else "num_of_products"
    card_col = "nums_card" if "nums_card" in df.columns else "credit_card"
    if prod_col in df.columns and card_col in df.columns:
        df["service_per_card"] = (
            df[prod_col] / df[card_col].replace(0, np.nan)
        ).fillna(df[prod_col])

    # Log-transform để giảm skewness
    for col in ["balance", "monthly_ir"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    print(f"[INFO] Feature Engineering: thêm {['balance_to_income', 'service_per_card', 'log_balance', 'log_monthly_ir']}")
    return df


# ---------------------------------------------------------------------------
# Drop Leakage & Identity
# ---------------------------------------------------------------------------

def drop_leakage_cols(df: pd.DataFrame, extra_cols: List[str] = None) -> pd.DataFrame:
    """
    Loại bỏ:
      1. Biến rò rỉ hậu kỳ: tenure_ye, engagement_score
         (2 biến này mã hóa 100% nhãn theo quy tắc nghiệp vụ tĩnh)
      2. Cột định danh: id, full_name, address, v.v.
      3. Các cột bổ sung do người dùng chỉ định (extra_cols).

    Args:
        df        : DataFrame đầu vào.
        extra_cols: Danh sách cột bổ sung cần bỏ.

    Returns:
        DataFrame đã loại bỏ các cột không hợp lệ.
    """
    to_drop = LEAKAGE_COLS + IDENTITY_COLS
    if extra_cols:
        to_drop += extra_cols

    existing = [c for c in to_drop if c in df.columns]
    df = df.drop(columns=existing, errors="ignore")
    print(f"[INFO] Đã loại bỏ {len(existing)} cột: {existing}")
    return df


# ---------------------------------------------------------------------------
# Encode categoricals
# ---------------------------------------------------------------------------

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuyển đổi biến phân loại sang số:
      - Cột nhị phân (2 giá trị): label encoding 0/1
      - Cột đa giá trị: one-hot encoding (drop_first=True)
      - Cột mục tiêu loyalty_level: map sang 0/1/2

    Returns:
        DataFrame với toàn bộ cột là số.
    """
    df = df.copy()

    # Encode nhãn mục tiêu
    if LABEL_COL in df.columns and df[LABEL_COL].dtype == object:
        df[LABEL_COL] = df[LABEL_COL].map(LABEL_MAP)
        print(f"[INFO] Đã encode '{LABEL_COL}': {LABEL_MAP}")

    # Tự động phát hiện và encode cột phân loại còn lại
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if LABEL_COL in cat_cols:
        cat_cols.remove(LABEL_COL)

    if cat_cols:
        binary_cols = [c for c in cat_cols if df[c].nunique() == 2]
        multi_cols = [c for c in cat_cols if df[c].nunique() > 2]

        for c in binary_cols:
            vals = df[c].unique()
            df[c] = df[c].map({vals[0]: 0, vals[1]: 1})

        if multi_cols:
            df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

        print(f"[INFO] Encode phân loại: binary={binary_cols}, one-hot={multi_cols}")

    return df


# ---------------------------------------------------------------------------
# VIF Check
# ---------------------------------------------------------------------------

def vif_check(
    X: pd.DataFrame,
    threshold: float = 5.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Kiểm tra Variance Inflation Factor (VIF) cho các biến liên tục.

    VIF_j = 1 / (1 - R²_j) đo mức độ một biến bị giải thích bởi các biến khác.
    VIF > 5 → đa cộng tuyến nghiêm trọng → có thể làm sai lệch trọng số hồi quy.

    Args:
        X        : DataFrame các feature liên tục đã StandardScaler.
        threshold: Ngưỡng cảnh báo (mặc định 5.0).

    Returns:
        DataFrame với cột ["feature", "VIF"] sắp xếp giảm dần.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_num = X[numeric_cols].copy().dropna()
    # Lấy mẫu đại diện tối đa 10,000 dòng để tính VIF nhanh chóng mà vẫn chính xác
    if len(X_num) > 10000:
        X_sample = X_num.sample(10000, random_state=42).values
    else:
        X_sample = X_num.values

    vif_data = pd.DataFrame({
        "feature": numeric_cols,
        "VIF": [variance_inflation_factor(X_sample, i)
                for i in range(len(numeric_cols))],
    }).sort_values("VIF", ascending=False).reset_index(drop=True)

    if verbose:
        print("\n[VIF Check] Variance Inflation Factor:")
        print(vif_data.to_string(index=False))
        high_vif = vif_data[vif_data["VIF"] > threshold]
        if len(high_vif) > 0:
            print(f"\n⚠️  CẢNH BÁO: {len(high_vif)} biến có VIF > {threshold}:")
            print(high_vif[["feature", "VIF"]].to_string(index=False))
        else:
            print(f"\n✅ Tất cả biến có VIF ≤ {threshold} — không có đa cộng tuyến nghiêm trọng.")

    return vif_data


# ---------------------------------------------------------------------------
# Stratified Split
# ---------------------------------------------------------------------------

def split_stratified(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple:
    """
    Chia tập dữ liệu theo tỷ lệ 70/15/15 với Stratified Sampling.

    Đảm bảo phân phối lớp Gold (2.2%) đồng đều trong Train/Val/Test.

    Args:
        X        : Feature matrix.
        y        : Nhãn (int).
        val_size : Tỷ lệ tập validation (mặc định 0.15).
        test_size: Tỷ lệ tập test (mặc định 0.15).
        seed     : Random seed.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    test_ratio = test_size
    val_ratio_of_trainval = val_size / (1.0 - test_size)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio_of_trainval,
        stratify=y_trainval,
        random_state=seed,
    )

    for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        n = len(y_split)
        dist = {k: f"{(y_split == k).sum()/n:.1%}" for k in [0, 1, 2]}
        print(f"  {split_name:5s}: {n:6d} mẫu | Bronze={dist[0]} Silver={dist[1]} Gold={dist[2]}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    raw_path: str,
    processed_dir: str = None,
    seed: int = 42,
) -> dict:
    """
    Chạy toàn bộ pipeline tiền xử lý từ đầu đến cuối.

    Thứ tự:
      1. Load dữ liệu thô
      2. Feature engineering
      3. Drop leakage & identity cols
      4. Encode categoricals
      5. StandardScaler (fit trên train, transform trên val/test)
      6. VIF check
      7. Stratified split 70/15/15
      8. Lưu processed data (nếu processed_dir được chỉ định)

    Returns:
        Dict chứa X_train, X_val, X_test, y_train, y_val, y_test,
        feature_names, scaler.
    """
    print("\n" + "="*60)
    print("  PIPELINE TIỀN XỬ LÝ DỮ LIỆU")
    print("="*60)

    # 1. Load
    df = load_raw_data(raw_path)

    # 2. Feature Engineering
    df = feature_engineering(df)

    # 3. Drop leakage & identity
    df = drop_leakage_cols(df)

    # 4. Encode categoricals
    df = encode_categoricals(df)
    df = df.dropna(subset=[LABEL_COL])
    df = df.dropna()

    # Tách X, y
    y = df[LABEL_COL].values.astype(int)
    X_df = df.drop(columns=[LABEL_COL])
    feature_names = list(X_df.columns)
    X = X_df.values

    # 5. Stratified split (raw, trước khi scale)
    print("\n[INFO] Chia tập dữ liệu:")
    X_train, X_val, X_test, y_train, y_val, y_test = split_stratified(
        X, y, seed=seed
    )

    # 6. StandardScaler (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 7. VIF check trên X_train đã scale
    print("\n[INFO] Kiểm tra VIF trên tập Train (sau StandardScaler):")
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    vif_df = vif_check(X_train_df, threshold=5.0)

    # 8. Lưu nếu cần
    if processed_dir:
        out_dir = Path(processed_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "X_train.npy", X_train)
        np.save(out_dir / "X_val.npy", X_val)
        np.save(out_dir / "X_test.npy", X_test)
        np.save(out_dir / "y_train.npy", y_train)
        np.save(out_dir / "y_val.npy", y_val)
        np.save(out_dir / "y_test.npy", y_test)
        import json as _json
        with open(out_dir / "feature_names.json", "w", encoding="utf-8") as _f:
            _json.dump(feature_names, _f)
        pd.DataFrame({"feature": feature_names}).to_csv(out_dir / "feature_names.csv", index=False)
        print(f"\n[INFO] Đã lưu processed data vào: {out_dir}")

    print("\n✅ Pipeline hoàn tất!")
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
        "vif_df": vif_df,
    }


if __name__ == "__main__":
    import sys
    raw = (
        sys.argv[1]
        if len(sys.argv) > 1
        else r"data/raw/Bank churn data set 80k.xlsx"
    )
    result = run_full_pipeline(
        raw_path=raw,
        processed_dir="data/processed",
    )
    print(f"\nShape Train: {result['X_train'].shape}")
    print(f"Shape Val  : {result['X_val'].shape}")
    print(f"Shape Test : {result['X_test'].shape}")

