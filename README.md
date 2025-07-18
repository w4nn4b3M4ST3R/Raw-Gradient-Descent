# 🧠 Raw ML Model: Build & Train from Scratch

This repository showcases how core machine learning models can be implemented and trained **manually from scratch** — without using frameworks like scikit-learn or TensorFlow. The focus is on understanding internal mechanisms and optimization logic through code.

---

## 📂 Modules in This Project

### 🔹 1. All Linear Regression Model

- Multi-variable model trained using basic numpy operations
- Manual weight & bias updates via MSE gradient
- Includes visualization of loss and predictions

### 🔹 2. Feature Engineering & Preprocessing

- Outlier removal (IQR method)
- Skewness & kurtosis analysis
- OneHotEncoding without sklearn
- Log transform & scaling options

### 🔹 3. Performance Metrics

- MAE, MSE, RMSE, MAPE, R²
- Manual implementation and comparisons
- Baseline vs. learned model performance

### 🔹 4. Exploratory Data Analysis (EDA)

- Seaborn visualizations
- Statistical summaries
- Correlation heatmaps

---

## 📊 Dataset

Current demos use:

- **Ad Spending vs. Product Sales** dataset
- Other datasets can be plugged in for regression tasks

---

## 💡 Why Raw ML?

This project is designed to help you:

- See what goes on **beneath the surface** of ML libraries
- Build intuition for optimization & loss
- Understand why models fail or succeed
- Transition smoothly into advanced ML workflows

---

## 🚀 How to Run

1. Clone the repo and open `raw_model.ipynb`
2. Customize hyperparameters like `learning_rate` and `max_epoch`
3. Run all cells and follow training visualization

> Recommend using Google Colab for smoother execution

---
