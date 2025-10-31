import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = 0
        self.b = 0

    def fit(self, X, y):
        for i in range(self.iterations):
            m = len(y)
            predictions = X.dot(self.w) + self.b
            d_dw = (1/m) * np.sum((predictions - y) * X)
            d_db = (1/m) * np.sum(predictions - y)
            self.w = self.w - (self.learning_rate * d_dw)
            self.b = self.b - (self.learning_rate * d_db)

    def predict(self, X):
        predictions = X.dot(self.w) + self.b
        return predictions
