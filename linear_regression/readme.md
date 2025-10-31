# 📘 Linear Regression with Gradient Descent (with L2 Regularization)

This project demonstrates how to implement **Linear Regression** from scratch using **Gradient Descent** optimization and includes support for **L2 Regularization** (Ridge Regression). It also visualizes the training process and evaluates model performance.

---

## 🧮 Mathematical Background

### 1. Linear Regression Model

Linear regression assumes a linear relationship between input features ( X \in \mathbb{R}^{m \times n} ) and output variable ( y \in \mathbb{R}^m ).

The hypothesis function is given by:

[
\hat{y}^{(i)} = w^T x^{(i)} + b
]

Where:

* ( \hat{y}^{(i)} ) is the predicted output for sample ( i )
* ( w \in \mathbb{R}^n ) is the weight vector
* ( b \in \mathbb{R} ) is the bias (intercept)
* ( x^{(i)} \in \mathbb{R}^n ) is the feature vector for the ( i^{th} ) sample

---

### 2. Cost Function (Mean Squared Error)

To measure model performance, the **Mean Squared Error (MSE)** cost function is used:

[
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
]

where:

* ( m ) = number of training examples
* ( y^{(i)} ) = true label for sample ( i )

---

### 3. L2 Regularization (Ridge Regression)

To prevent overfitting and penalize large weight magnitudes, we add an **L2 regularization** term:

[
J_{reg}(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
]

where:

* ( \lambda ) = regularization parameter controlling penalty strength
* Larger ( \lambda ) ⇒ stronger regularization ⇒ smaller weights (but possibly higher bias)

---

### 4. Gradient Descent Optimization

We minimize the cost function ( J_{reg}(w, b) ) iteratively using **Gradient Descent**.

The parameter update rules are derived from the partial derivatives:

[
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \big( (\hat{y}^{(i)} - y^{(i)})x_j^{(i)} \big) + \frac{\lambda}{m}w_j
]

[
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
]

Hence, updates become:

[
w_j := w_j - \alpha \frac{\partial J}{\partial w_j}
]
[
b := b - \alpha \frac{\partial J}{\partial b}
]

where ( \alpha ) is the **learning rate** controlling the step size in gradient descent.

---

### 5. Convergence Criterion

Training stops when the **absolute change in cost** between iterations is smaller than a threshold ( \epsilon ):

[
|J_{t} - J_{t-1}| \leq \epsilon
]

This ensures that the optimization terminates once the model converges.

---

## ⚙️ Implementation Details

The code consists of two main parts:

1. **Procedural Implementation** (for intuition)
2. **Object-Oriented Implementation** via the `LinearRegression` class

### Key Parameters

| Parameter       | Description                            | Default   |
| --------------- | -------------------------------------- | --------- |
| `learning_rate` | Step size for gradient descent updates | 0.001     |
| `epsilon`       | Convergence threshold                  | 1e-6      |
| `_lambda`       | Regularization parameter               | 0         |
| `max_iter`      | Maximum number of training iterations  | 1,000,000 |

---

## 🧠 Class Overview

```python
class LinearRegression:
    def __init__(self, learning_rate=0.001, epsilon=1e-6, _lambda=0, max_iter=1000000)
```

### Methods

#### 1. `fit(X, y)`

Trains the model using gradient descent.

Steps:

1. Initialize weights and bias to zero.
2. Compute predictions:
   [
   \hat{y} = Xw + b
   ]
3. Compute cost using `_cost_function`.
4. Compute gradients for `w` and `b`.
5. Update parameters iteratively until convergence.

#### 2. `_cost_function(X, y, w, b)`

Computes the mean squared error (and regularization term if `_lambda` > 0).

#### 3. `predict(X)`

Generates predictions using the learned weights and bias.

---

## 📊 Visualizations

Two plots are generated:

1. **Cost Function History**

   * Shows how the cost decreases over iterations.
   * Helps visualize convergence behavior.

2. **Regression Line Fit**

   * Plots data points and fitted line (for 1D case).
   * Demonstrates the quality of the fit.

---

## 🧩 Example Usage

```python
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

The **Mean Squared Error (MSE)** is used to measure performance:

[
MSE = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})^2
]

and the **Root Mean Squared Error (RMSE)** provides an interpretable error in the same unit as ( y ):

[
RMSE = \sqrt{MSE}
]

---

## 📈 Typical Output

```
Weights: [ 1.98 -2.99]
Bias: 5.01
MSE: 0.035
RMSE: 0.187
```

This demonstrates that the model successfully recovered the true underlying parameters (( w = [2, -3], b = 5 )).

---

## 🧩 Notes

* The algorithm is **batch gradient descent**, i.e., it uses all data in each update.
* You can extend this to **stochastic** or **mini-batch** versions for large datasets.
* Regularization can be turned off by setting `_lambda = 0`.

---

## 🧠 Summary

| Concept                | Formula / Description                                                                                            |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Hypothesis             | ( \hat{y} = Xw + b )                                                                                             |
| Cost Function          | ( J = \frac{1}{2m}\sum(\hat{y}-y)^2 + \frac{\lambda}{2m}\sum w^2 )                                               |
| Gradient w.r.t weights | ( \frac{1}{m}\sum((\hat{y}-y)X) + \frac{\lambda}{m}w )                                                           |
| Gradient w.r.t bias    | ( \frac{1}{m}\sum(\hat{y}-y) )                                                                                   |
| Update Rule            | ( w := w - \alpha \cdot \frac{\partial J}{\partial w} ), ( b := b - \alpha \cdot \frac{\partial J}{\partial b} ) |

---

## 🧰 Dependencies

```bash
pip install numpy matplotlib seaborn
```

---

## 📚 References

* *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron
* Andrew Ng’s *Machine Learning* (Coursera)

