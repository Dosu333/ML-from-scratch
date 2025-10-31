import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.001, epsilon=1e-6,
                 _lambda=0, max_iter=1000000):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self._lambda = _lambda
        self.max_iter = max_iter
        self.w = None
        self.b = 0

    def _cost_function(self, X, y, w, b):
        m = len(y)
        predictions = X.dot(w) + b
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        y = y.ravel()
        prev_cost = None
        new_cost = None

        for _ in range(self.max_iter):
            prev_cost = new_cost
            new_cost = self._cost_function(X, y, self.w, self.b)

            if (prev_cost is not None and
                    abs(new_cost - prev_cost) <= self.epsilon):
                break

            m = len(y)
            predictions = X.dot(self.w) + self.b
            error = predictions - y

            for j in range(X.shape[1]):
                d_dw = (1 / m) * np.sum(error * X[:, j]) + (
                    (self._lambda / m) * self.w[j]
                )
                self.w[j] -= self.learning_rate * d_dw
            d_db = 1 / m * np.sum(error)
            self.b -= self.learning_rate * d_db

    def predict(self, X):
        predictions = X.dot(self.w) + self.b
        return predictions
