import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.w = 0
        self.b = 0

    def _cost_function(self, X, y, w, b):
        m = len(y)
        predictions = X.dot(w) + b
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    def fit(self, X, y):
        prev_cost = None
        new_cost = None

        while prev_cost is None or abs(new_cost - prev_cost) > self.epsilon:
            m = len(y)
            predictions = X.dot(self.w) + self.b
            prev_cost = new_cost
            new_cost = self._cost_function(X, y, self.w, self.b)
            d_dw = (1 / m) * np.sum((predictions - y) * X)
            d_db = (1 / m) * np.sum(predictions - y)
            self.w = self.w - (self.learning_rate * d_dw)
            self.b = self.b - (self.learning_rate * d_db)

    def predict(self, X):
        predictions = X.dot(self.w) + self.b
        return predictions
