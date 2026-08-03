# Classification & Regression Algorithms from Scratch (OLS, GD & Regularization)
> Implementing foundational machine learning algorithms from scratch—without external ML libraries—to explore optimization mechanics, closed-form matrix solvers, and L1/L2 regularization.

---

## 📌 Project Overview

This project focuses on the mathematical mechanics and implementation of foundational classification and regression algorithms from the ground up, strictly avoiding high-level machine learning APIs such as Scikit-Learn. 

- **Optimization Dynamics:** Implementing custom **Gradient Descent** loops and activation functions (e.g., Sigmoid) for probabilistic binary classification.
- **Closed-Form Matrix Computations:** Deriving and executing **Ordinary Least Squares (OLS)** solutions for linear and regularized regression models.
- **Regularization Trade-offs:** Evaluating how L1 (Lasso) and L2 (Ridge) penalties mitigate overfitting across noisy, real-world tabular datasets.

---

## 🛠️ Tech Stack & Methodology

- **Languages & Core Libraries:** Python 3, NumPy, Pandas, Matplotlib
- **Data Preprocessing Pipeline:**
  - **Outlier Handling:** Implemented the **Interquartile Range (IQR)** threshold method to detect and filter out noisy sample rows across continuous features.
  - **Missing Value Imputation:** Applied median imputation for datasets with moderate missing values (~4.9% in the Penguins dataset) and row-dropping for large-scale datasets (Diamonds dataset).
  - **Categorical Encoding:** Converted string attributes (`species`, `island`, `gender`) into binary representations using **One-Hot Encoding**.
  - **Feature Scaling:** Applied custom **Min-Max Normalization** to scale continuous numerical variables into the $[0, 1]$ range for training stability.

---

## 📊 Experiments & Model Performance

Experiments were conducted across three distinct tabular datasets to evaluate model adaptability across classification and regression tasks.

| Dataset | Task Type | Implemented Algorithm | Target Variable | Key Results & Evaluation |
| :--- | :--- | :--- | :--- | :--- |
| **Penguins** | Binary Classification | `Logistic Regression (GD)` | `is_biscoe_island` (0 or 1) | Achieved 96.0% accuracy on the test set; demonstrated smooth convergence in loss vs. iteration plots. |
| **Diamonds** | Binary Classification & Regression | `Logistic Regression`, `Elastic Net (GD)` | `is_ideal_cut` / `price` | Effectively handled 53,940 rows; Elastic Net regularization reduced generalization error compared to baseline regression. |
| **Wine** | Binary Classification & Regression | `Linear (OLS)`, `Ridge (OLS)` | `is_red_wine` / `alcohol` | OLS Ridge penalty successfully constrained weight magnitudes without gradient step tuning. |

### Key Insights
- **OLS vs. Gradient Descent:** While closed-form OLS provides exact analytical solutions for Linear and Ridge regression without hyperparameter tuning, custom Gradient Descent was essential for Elastic Net where non-differentiable L1 penalties require iterative optimization.
- **Impact of Preprocessing:** Min-Max normalization across numerical attributes significantly accelerated gradient convergence in Logistic Regression and Elastic Net.

---

## 💡 Troubleshooting & Learnings

- **Handling Real-World Noisy Data:** Managing mixed string formats, outliers, and missing entries across features required building a robust, reusable preprocessing pipeline. Stripping outliers via IQR before normalization prevented extreme values from compressing the distribution of standard features.
- **Hyperparameter Sensitivity in Elastic Net:** Experiencing gradient divergence during early Elastic Net trials underscored the importance of learning rate decay and proper initialization of weights, leading to more stable optimization.
- **Matrix Singularity in OLS:** Observed that highly correlated one-hot encoded columns could lead to non-invertible matrices during Ordinary Least Squares computation; adding a small Ridge regularization term resolved singularity issues effectively.
