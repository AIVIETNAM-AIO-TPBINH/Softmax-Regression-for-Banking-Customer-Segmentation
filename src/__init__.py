"""
Robust & Calibrated Softmax Regression for Banking Customer Segmentation
=========================================================================
Các module chính:
  - preprocessing : Tải, làm sạch và kỹ nghệ đặc trưng
  - losses        : Hàm mất mát (Logit Adjustment, ε-Softmax)
  - noise         : Tạo nhiễu nhãn liền kề có kiểm soát
  - metrics       : Bộ đo lường đầy đủ (ECE, PR-AUC, Balanced Acc)
  - models        : Classifier sklearn-compatible với loss tùy biến
  - calibration   : Temperature Scaling (L-BFGS-B)
  - explainability: Phân tích Odds Ratio
  - utils         : Hàm tiện ích (seed, plot, save)
"""

