import matplotlib.pyplot as plt
import numpy as np

X = np.array([30, 40, 50, 60, 70, 80, 90, 100, 110, 120], dtype=np.float64)
Y = np.array(
    [145, 213, 240, 320, 340, 420, 470, 510, 560, 600], dtype=np.float64
)


class GradientDescent:
    def __init__(self, learning_rate=1e-4, num_epochs=1000):
        self.lr = learning_rate
        self.epoch = num_epochs

    def initial(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0

    def fit(self, X, Y):
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_vals, n_features = X.shape
        self.initial(n_features)
        for epoch in range(self.epoch):
            Y_hat = np.dot(X, self.w) + self.b
            delta_Y = Y_hat - Y

            dw = 2 * np.dot(X.T, delta_Y) / n_vals
            db = 2 * np.sum(delta_Y) / n_vals

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return [np.dot(X, self.w) + self.b]

    def get_params(self):
        return (self.w, self.b)


gb_model = GradientDescent()
gb_model.fit(X, Y)
y_predict = gb_model.predict(X)


plt.figure(figsize=(8, 6))
plt.scatter(y_predict, Y, label="Predicted vs Actual")
plt.plot(Y, Y, label="Perfect prediction line", color="red")
plt.xlabel("Prediction")
plt.ylabel("Reality")
plt.legend()
plt.grid(True)
plt.title("Prediction vs Reality")
plt.show()
