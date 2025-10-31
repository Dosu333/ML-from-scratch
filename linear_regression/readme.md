# 📘 Linear Regression with Gradient Descent (with L2 Regularization)

This project implements **Linear Regression** from scratch using **Gradient Descent**, with optional **L2 Regularization** (Ridge Regression). It also visualizes the training process and evaluates model performance.

---

## 🧮 Mathematical Background

### 1. Linear Regression Model

We model the relationship between input features `X` and target `y` as a linear combination of weights and bias:

```
ŷ = wᵀx + b
```

Where:

* ŷ = predicted value
* w = weights vector
* b = bias (intercept)
* x = feature vector for one training example

---

### 2. Cost Function (Mean Squared Error)

The Mean Squared Error (MSE) measures how well predictions match actual values:

```
J(w, b) = (1 / (2m)) * Σ(ŷᵢ - yᵢ)²
```

Where:

* m = number of training examples
* yᵢ = actual target
* ŷᵢ = predicted value for sample i

---

### 3. L2 Regularization (Ridge Regression)

To penalize large weight values and reduce overfitting, a regularization term is added:

```
J_reg(w, b) = (1 / (2m)) * Σ(ŷᵢ - yᵢ)² + (λ / (2m)) * Σwⱼ²
```

Where:

* λ (lambda) = regularization strength
* Larger λ → stronger penalty → smaller weights

---

### 4. Gradient Descent Optimization

To minimize the cost function, we iteratively update parameters using gradients:

```
∂J/∂wⱼ = (1/m) * Σ((ŷᵢ - yᵢ) * xⱼᵢ) + (λ/m) * wⱼ
∂J/∂b  = (1/m) * Σ(ŷᵢ - yᵢ)
```

Update rules:

```
wⱼ := wⱼ - α * (∂J/∂wⱼ)
b  := b  - α * (∂J/∂b)
```

Where:

* α (alpha) = learning rate controlling step size

---

### 5. Convergence Criterion

Training stops when the change in cost between iterations is smaller than a threshold ε (epsilon):

```
|J(t) - J(t-1)| ≤ ε
```

---

## ⚙️ Implementation Details

The project includes both:

1. A **procedural** implementation for intuition
2. An **object-oriented** version (`LinearRegression` class)

### Key Parameters

| Parameter       | Description                            | Default   |
| --------------- | -------------------------------------- | --------- |
| `learning_rate` | Step size for gradient descent updates | 0.001     |
| `epsilon`       | Convergence threshold                  | 1e-6      |
| `_lambda`       | Regularization parameter               | 0         |
| `max_iter`      | Maximum number of iterations           | 1,000,000 |

---

## 🧠 Class Overview

```python
class LinearRegression:
    def __init__(self, learning_rate=0.001, epsilon=1e-6, _lambda=0, max_iter=1000000)
```

### Methods

#### `fit(X, y)`

Trains the model using batch gradient descent:

1. Initialize weights and bias to zero.
2. Compute predictions: ŷ = Xw + b
3. Compute cost using `_cost_function`.
4. Compute gradients for w and b.
5. Update parameters iteratively until convergence or max iterations.

#### `_cost_function(X, y, w, b)`

Computes the mean squared error with optional regularization.

#### `predict(X)`

Generates predictions using the learned weights and bias.

---

## 📊 Visualizations

Two key plots help track model performance:

1. **Cost Function History** — shows how the cost decreases over iterations.
2. **Regression Line Fit** — displays the data points and regression line (for 1D case).

---

## 🧩 Example Usage

```python
import numpy as np

np.random.seed(42)
X = np.random.rand(100, 2)
true_w = np.array([2, -3])
y = X.dot(true_w) + 5 + np.random.randn(100) * 0.2 

model = LinearRegression(learning_rate=0.01)
model.fit(X, y)

print("Weights:", model.w)
print("Bias:", model.b)
y_pred = model.predict(X)

mse = np.mean((y - y_pred) ** 2)
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))
```

---

## 🧾 Evaluation Metrics

We use **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)** to evaluate model performance:

```
MSE  = (1/m) * Σ(ŷᵢ - yᵢ)²
RMSE = sqrt(MSE)
```

---

## 📈 Typical Output

```
Weights: [ 1.98 -2.99]
Bias: 5.01
MSE: 0.035
RMSE: 0.187
```

This shows that the model successfully learned parameters close to the true values (w = [2, -3], b = 5).

---

## 🧩 Notes

* The algorithm uses **batch gradient descent** (all samples per update).
* You can extend it to **stochastic** or **mini-batch** versions for large datasets.
* Regularization can be disabled by setting `_lambda = 0`.

---

## 🧠 Summary

| Concept       | Description                               |
| ------------- | ----------------------------------------- |
| Hypothesis    | ŷ = Xw + b                                |
| Cost Function | J = (1 / 2m) * Σ(ŷ - y)² + (λ / 2m) * Σw² |
| Gradient (w)  | (1/m) * Σ((ŷ - y)X) + (λ/m) * w           |
| Gradient (b)  | (1/m) * Σ(ŷ - y)                          |
| Update Rule   | w := w - α * ∂J/∂w; b := b - α * ∂J/∂b    |

---

## 🧰 Dependencies

```bash
pip install numpy matplotlib seaborn
```

---

## 📚 References

* *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron
* Andrew Ng’s *Machine Learning* (Coursera)

