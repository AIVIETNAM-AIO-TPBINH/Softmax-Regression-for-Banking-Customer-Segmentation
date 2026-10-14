# Robust & Calibrated Softmax Regression for Banking Customer Segmentation

Nghiên cứu thực nghiệm bóc tách (Ablation Study) về giải pháp xử lý mất cân bằng dữ liệu đa lớp, kháng nhiễu nhãn và hiệu chuẩn xác suất sử dụng Softmax Regression trên tập dữ liệu 80,000 khách hàng ngân hàng.

## 📌 Tổng Quan
- **Subtopic:** Softmax Regression (Deep Learning Module - AIO2026).
- **Bộ dữ liệu:** Bank Churn Dataset (80,000 bản ghi, 26 thuộc tính).
- **Bài toán:** Phân loại đa lớp hạng thành viên (`loyalty_level`: Bronze 86.6%, Silver 11.2%, Gold 2.2%).

## 🛠️ Phương Pháp Triển Khai
1. **Logit-Adjusted Loss (ICLR 2021):** Can thiệp logit theo tần suất tiên nghiệm để phục hồi Recall cho lớp thiểu số (Gold).
2. **ε-Softmax (NeurIPS 2024):** Thay thế phân phối one-hot chuẩn để tăng tính kháng nhiễu nhãn (Label Noise).
3. **Temperature Scaling (ICML 2017):** Hiệu chuẩn xác suất đầu ra sau huấn luyện để triệt tiêu hiện tượng tự tin thái quá (Overconfidence) và tối ưu chỉ số ECE.

## 🚀 Hướng Dẫn Tái Lập Thí Nghiệm
```bash
# 1. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 2. Chạy pipeline tiền xử lý dữ liệu (Feature Engineering & Stratified Split)
python -m src.preprocessing

# 3. Chạy các Notebook thực nghiệm
jupyter notebook notebooks/01_eda_and_preprocessing.ipynb
jupyter notebook notebooks/02_baseline_model.ipynb
jupyter notebook notebooks/03_module_implementations.ipynb
jupyter notebook notebooks/04_ablation_experiments.ipynb
```

## 📁 Cấu Trúc Dự Án
```
.
├── data/
│   ├── raw/                 # Dữ liệu gốc (Bank churn data set 80k.xlsx)
│   └── processed/           # Dữ liệu sau tiền xử lý (Train/Val/Test numpy arrays)
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_module_implementations.ipynb
│   └── 04_ablation_experiments.ipynb
├── plans/                   # Kế hoạch nghiên cứu chính thức (.docx)
├── reports/
│   ├── WEEK04_Final_Technical_Report.md  # Báo cáo kỹ thuật toàn văn
│   ├── figures/             # Biểu đồ phân tích độ phân giải cao
│   ├── 02_baselines_results.json
│   └── ablation_results.json
├── src/
│   ├── calibration.py       # Temperature Scaling
│   ├── explainability.py    # Phân tích Odds Ratio
│   ├── losses.py            # Logit Adjustment & ε-Softmax
│   ├── metrics.py           # Bộ đo lường (Balanced Acc, Macro F1, ECE...)
│   ├── models.py            # LogitAdjustedClassifier & EpsilonSoftmaxClassifier
│   ├── noise.py             # Giả lập nhiễu nhãn liền kề (Adjacent Label Noise)
│   ├── preprocessing.py     # Pipeline tiền xử lý hoàn chỉnh
│   └── utils.py             # Tiện ích trực quan hóa & I/O
├── requirements.txt
└── README.md
```