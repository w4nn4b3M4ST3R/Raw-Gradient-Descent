import matplotlib.pyplot as plt
import numpy as np

X = np.array([30, 40, 50, 60, 70, 80, 90, 100, 110, 120], dtype=np.float64)
Y = np.array(
    [145, 213, 240, 320, 340, 420, 470, 510, 560, 600], dtype=np.float64
)


class GradientDescent:
    def __init__(self, w0=0, b0=0, learning_rate=1e-4, num_epochs=1000):
        self.w = w0
        self.b = b0
        self.lr = learning_rate
        self.epoch = num_epochs
        self.X = None
        self.Y = None
        self.len = None

    def fit(self, X, Y):
        self.X = np.array(X, dtype=np.float64)
        self.Y = np.array(Y, dtype=np.float64)
        self.len = len(X)
        self.update()

    def update(self):
        for epoch in range(self.epoch):
            for i in range(self.len):
                x, y = self.X[i], self.Y[i]

                y_hat = self.w * x + self.b
                delta_y = y_hat - y

                dw = 2 * x * delta_y / self.len
                db = 2 * delta_y / self.len

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X: np.ndarray):
        return [self.w * x + self.b for x in X]

    def get_params(self):
        return (self.w, self.b)

    def create_plot(self):
        y_predict = self.predict(self.X)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_predict, self.Y, label="Predicted vs Actual")
        plt.plot(self.Y, self.Y, label="Perfect prediction line", color="red")
        plt.xlabel("Prediction")
        plt.ylabel("Reality")
        plt.legend()
        plt.grid(True)
        plt.title("Prediction vs Reality")
        plt.show()


gb_model = GradientDescent()
gb_model.fit(X, Y)
gb_model.create_plot()
