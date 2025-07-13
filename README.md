# 📈 Manual Gradient Descent Linear Regression

This project implements a **multi-variable linear regression model** trained entirely by hand using **gradient descent** — without any machine learning frameworks. It's designed to help learners and developers understand core optimization logic step-by-step.

---

## 📊 Sample Dataset: Predict Sales from Ad Spending

This dataset is used to train the regression model to predict product sales based on budget allocations for advertising in three channels: TV, Radio, and Newspaper.

---

## 🧠 Concept

### We model a prediction function:

- TV : x1
- Radio: x2
- Newspaper: x3
- Sales: y

`ŷ = w₁·x₁ + w₂·x₂ + w₃·x₃ + b`

Weights `w1`, `w2`, `w3`, and bias `b` are updated manually using the gradients of Mean Squared Error (MSE) loss:

```python
loss = 0.5 * (y_hat - y)**2

dw1 = x1 * (y_hat - y)
dw2 = x2 * (y_hat - y)
dw3 = x3 * (y_hat - y)
db  =      (y_hat - y)

```

---

## 🚀 How to Use

- 1. Run the notebook: `gradient_descent.ipynb` on local or Google Colab _(highly recommended)_

- 2. Set learning rate and max epoch

_Default:_

```python
lr = 1e-5
max_epoch = 1000
```

- 3. `Run all`
