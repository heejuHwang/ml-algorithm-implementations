# Machine Learning & Deep Learning Implementations

A comprehensive archive of foundational to advanced Machine Learning, Deep Learning, and Reinforcement Learning algorithms. This repository emphasizes **building models from the ground up**, understanding mathematical mechanics, and systematic **performance tuning and evaluation** across various datasets.

---

## 📚 Master Table of Projects & Implementations

### 1. Supervised Learning: Regression & Classification
| Project / Domain | Implemented Algorithms | Key Tech / Methods | Summary of Results | View |
| :--- | :--- | :--- | :--- | :---: |
| **Multi Dataset Regression** | `Logistic Regression`, `Linear`, `Logistic`, `ElasticNet` | EDA, L1/L2 , Gradient Descent, OLS, IQR Outlier Removal, One-Hot Encoding | Achieved **96.0% accuracy** on Penguins classification; successfully prevented overfitting via L1/L2 penalties | [🔗 Code](./01-supervised-learning/multi-dataset-regression) |

### 2. Deep Learning: Computer Vision & Vision Models
| Project / Domain | Implemented Algorithms | Key Tech / Methods | Summary of Results | View |
| :--- | :--- | :--- | :--- | :---: |
| **PyTorch Neural Networks to VGG-13** | `Custom NN`, `CNN`, `VGG-13` | PyTorch from scratch, Batch Norm, Early Stopping, LR Scheduling, EMNIST (36 classes) | Achieved 92.0% accuracy on EMNIST test set; demonstrated trade-offs of adapting VGG-13 to $28 \times 28$ image constraints | [🔗 Code](./02-deep-learning/pytorch-neural-networks-to-vgg13) |


### 3. Reinforcement Learning
| Project / Domain | Implemented Algorithms | Key Tech / Methods | Summary of Results | View |
| :--- | :--- | :--- | :--- | :---: |

---

## 📁 Repository Structure

```text
ml-algorithm-implementations/
├──01-supervised-learning/
│   └── multi-dataset-regression/
│       ├── 01_data_preprocessing_and_eda.ipynb
│       ├── 02_logistic_regression.ipynb
│       ├── 03_ols_linear_and_ridge_regression_wine.ipynb
│       └── 04_elasticnet_gradient_descent_diamonds.ipynb
├──02-deep-learning/
│   └── pytorch-neural-networks-to-vgg13
│       ├── 01_binary_classification_nn.ipynb
│       ├── 02_binary_classification_nn_optimization.ipynb
│       ├── 03_emnist_cnn_classification.ipynb
│       └── 04_emnist_vgg13_adaptation.ipynb