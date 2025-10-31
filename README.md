# 🧮 Machine Learning Algorithms from Scratch (NumPy)

This repository is a collection of **machine learning algorithms implemented entirely from scratch using NumPy** — no Scikit-Learn, no TensorFlow.  
Each implementation focuses on understanding **how the math works under the hood**, with visualizations, derivations, and performance evaluation.

---

## 📚 Algorithms Implemented

### 1️⃣ [Linear Regression](./linear_regression/)
**Goal:** Predict continuous outcomes by minimizing Mean Squared Error (MSE).

**Highlights:**
- Built **from scratch** using gradient descent.  
- Includes **cost function derivation**, learning rate tuning, and convergence visualization.  
- Evaluated with **R² score** and **MSE**.  
- Visualizes regression line fit vs. data.

📘 [Read full details →](./linear_regression/readme.md)

---

### 2️⃣ [Logistic Regression](./logistic_regression/)
**Goal:** Classify data points into binary categories using the sigmoid function.

**Highlights:**
- Implements **binary cross-entropy loss** and **L2 regularization**.  
- Uses **gradient descent** for optimization.  
- Visualizes **decision boundary**, **sigmoid function**, and **cost convergence**.  
- Evaluated with **Precision** and **Recall** metrics.

📘 [Read full details →](./logistic_regression/readme.md)

---

## 🧩 Repo Structure

```

📂 ML-ALGO
┣ 📁 Linear_Regression
┃ ┣ 📜 linear_regression.ipynb
┃ ┗ 📜 README.md
┣ 📁 Logistic_Regression
┃ ┣ 📜 logistic_regression.ipynb
┃ ┗ 📜 README.md
┣ 📜 requirements.txt
┗ 📜 README.md   ← (this file)

```

---

## 🛠️ Technologies Used
- **Python 3.11+**
- **NumPy** — numerical operations  
- **Matplotlib** — plotting  
- **Seaborn / Pandas** — data visualization  
- *(No machine learning frameworks used)*  

---

## 📊 Visuals Included
Each algorithm notebook includes:
- Loss/cost function convergence plots  
- Model fit visualization  
- Sigmoid / Decision boundary plots (for logistic regression)

---

## 🔬 Learning Outcomes
By exploring these implementations, you’ll learn:
- How gradient descent works step by step  
- How to derive cost and gradient functions  
- How to visualize convergence and model performance  
- How regularization affects model training  

---

## 🌱 Next Steps
Upcoming algorithms to be added:
- [ ] Decision Tree
- [ ] K-Means 
- [ ] PCA  


---

## 🧑‍💻 Author
**Oladosu Larinde**  
Lead Software Engineer
💡 Passionate about building intelligent systems and teaching others how they work.

---

## 🌟 Inspiration
This project was inspired by:
- Andrew Ng’s ML Course (Coursera)  
- “Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow” by Aurélien Géron
