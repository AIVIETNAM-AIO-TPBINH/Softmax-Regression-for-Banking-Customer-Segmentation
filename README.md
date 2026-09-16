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
pip install -r requirements.txt
python -m src.preprocessing
jupyter notebook "Working Files/notebooks/04_ablation_experiments.ipynb"