# 🧠 Logistic Regression from Scratch (with L2 Regularization)

This project implements **Logistic Regression** from scratch using **Gradient Descent** optimization and optional **L2 Regularization** (Ridge). It demonstrates how a linear model can perform binary classification by learning probabilities through the **sigmoid function**.

---

## 🧮 Mathematical Background

### 1. Logistic Model (Hypothesis Function)

Logistic regression predicts the probability that a sample belongs to class 1:

```
ŷ = σ(z) = 1 / (1 + exp(-z))
```

where:

```
z = wᵀx + b
```

* ŷ — predicted probability of class 1
* w — weight vector
* b — bias term
* x — feature vector
* σ(z) — sigmoid function mapping real numbers to (0, 1)

---

### 2. Decision Rule

The predicted class is determined by thresholding the probability:

```
if ŷ >= 0.5 → predict 1
else → predict 0
```

You can adjust the threshold for precision/recall trade-off.

---

### 3. Cost Function (Binary Cross-Entropy)

The cost function measures how far the predictions are from the true labels:

```
J(w, b) = -(1/m) * Σ [ yᵢ * log(ŷᵢ) + (1 - yᵢ) * log(1 - ŷᵢ) ]
```

where:

* m = number of training samples
* yᵢ = true label (0 or 1)
* ŷᵢ = predicted probability

---

### 4. L2 Regularization (Ridge)

To prevent overfitting, we add an L2 penalty term that discourages large weights:

```
J_reg(w, b) = J(w, b) + (λ / (2m)) * Σ wⱼ²
```

where:

* λ (lambda) controls the regularization strength (higher λ → stronger penalty).

---

### 5. Gradient Descent Optimization

To minimize the cost, gradients are computed and used to update parameters:

```
∂J/∂wⱼ = (1/m) * Σ((ŷᵢ - yᵢ) * xⱼᵢ) + (λ/m) * wⱼ
∂J/∂b  = (1/m) * Σ(ŷᵢ - yᵢ)
```

Parameter updates:

```
wⱼ := wⱼ - α * (∂J/∂wⱼ)
b  := b  - α * (∂J/∂b)
```

where:

* α = learning rate
* The loop continues until cost convergence or reaching max iterations.

---

### 6. Convergence Criterion

Training stops when the cost no longer changes significantly:

```
|J(t) - J(t-1)| ≤ ε
```

---

## ⚙️ Implementation Details

### Parameters

| Parameter       | Description                            | Default |
| --------------- | -------------------------------------- | ------- |
| `learning_rate` | Step size for gradient descent updates | 0.5     |
| `epsi1on`       | Convergence threshold                  | 1e-6    |
| `_lambda`       | Regularization parameter               | 0       |
| `max_iter`      | Maximum number of iterations           | 10,000  |

---

## 🧩 Class Overview

```python
class LogisticRegression:
    def __init__(self, learning_rate=0.5, epsi1on=1e-6, _lambda=0, max_iter=10000)
```

### Methods

#### `fit(X, y)`

Trains the model by iteratively:

1. Computing predictions ŷ = σ(Xw + b)
2. Evaluating cost using the binary cross-entropy loss
3. Computing gradients for each parameter
4. Updating weights and bias until convergence

#### `_cost_function(X, y, w, b)`

Calculates the cost (and includes the regularization term if `_lambda` > 0).

#### `predict_proba(X)`

Returns predicted probabilities between 0 and 1.

#### `predict(X, threshold=0.5)`

Returns binary class predictions (0 or 1) based on a probability threshold.

---

## 🧠 Step-by-Step Visualization

The project first visualizes:

1. The **feature space** with colors based on target classes (`sns.pairplot`).
2. The **predicted probabilities** using a color map to show how the sigmoid function separates classes.

---

## 📊 Example Usage

```python
import numpy as np

np.random.seed(42)
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, size=(100,))

model = LogisticRegression(_lambda=0.1)
model.fit(X, y)

# Predictions
y_pred_proba = model.predict_proba(X)
y_pred = model.predict(X, threshold=0.5)

print("Predicted probabilities:", y_pred_proba[:5])
print("Predicted classes:", y_pred[:5])
```

---

## 🧾 Model Evaluation

We can compute standard binary classification metrics:

```
True Positive  (TP): predicted 1, actual 1
True Negative  (TN): predicted 0, actual 0
False Positive (FP): predicted 1, actual 0
False Negative (FN): predicted 0, actual 1
```

From these:

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

These metrics describe how well the model distinguishes the two classes.

---

## 📈 Visualization Example

**Predicted probabilities heatmap:**

```python
plt.scatter(X[:, 0], X[:, 1], c=y_pred_proba, cmap='viridis')
plt.colorbar(label='Predicted Probability')
plt.xlabel('feature_0')
plt.ylabel('feature_1')
plt.title('Predicted Probabilities from Logistic Regression')
plt.show()
```

**Cost function convergence:**

```python
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.show()
```

---

## 📦 Dependencies

```bash
pip install numpy seaborn matplotlib pandas
```

---

## 🧩 Key Concepts Summary

| Concept          | Formula                                     |
| ---------------- | ------------------------------------------- |
| Sigmoid Function | σ(z) = 1 / (1 + exp(-z))                    |
| Hypothesis       | ŷ = σ(wᵀx + b)                              |
| Cost Function    | J = -(1/m) Σ[y log(ŷ) + (1 - y) log(1 - ŷ)] |
| Regularization   | (λ / 2m) Σw²                                |
| Gradient (w)     | (1/m) Σ((ŷ - y)x) + (λ/m)w                  |
| Gradient (b)     | (1/m) Σ(ŷ - y)                              |
