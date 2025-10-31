import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.5, epsi1on=1e-6):
        self.learning_rate = learning_rate
        self.epsi1on = epsi1on
        self.w = None
        self.b = None

    def _cost_function(self, X, y, w, b):
        m = len(y)
        z = X.dot(w) + b
        f_w_b = 1 / (1 + np.exp(-z))
        loss_fn = -y * np.log(f_w_b) - (1 - y) * np.log(1 - f_w_b)
        cost = 1 / m * np.sum(loss_fn)
        return cost

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0

        prev_cost = None
        new_cost = None

        while prev_cost is None or abs(prev_cost - new_cost) > 1e-6:
            m = len(y)
            prev_cost = new_cost
            new_cost = self._cost_function(X, y, self.w, self.b)

            z = X.dot(self.w) + self.b
            f_w_b = 1 / (1 + np.exp(-z))
            error = f_w_b - y

            for j in range(X.shape[1]):
                d_dw = 1 / m * np.sum((error * X[:, j]))
                self.w[j] -= self.learning_rate * d_dw
            d_db = 1 / m * np.sum(error)
            self.b -= self.learning_rate * d_db

    def predict_proba(self, X):
        z = X.dot(self.w) + self.b
        f_w_b = 1 / (1 + np.exp(-z))
        return f_w_b

    def predict(self, X, threshold=0.5):
        z = X.dot(self.w) + self.b
        f_w_b = 1 / (1 + np.exp(-z))
        return (f_w_b >= threshold).astype(int)
