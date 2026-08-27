# Machine Learning & Deep Learning Implementations

A comprehensive archive of foundational to advanced Machine Learning, Deep Learning, and Reinforcement Learning algorithms. This repository emphasizes **building models from the ground up**, understanding mathematical mechanics, and systematic **performance tuning and evaluation** across various datasets.

---

## 📚 Master Table of Projects & Implementations

### 1. Supervised Learning: Regression & Classification
| Project / Domain | Implemented Algorithms | Key Tech / Methods | Summary of Results | View |
| :--- | :--- | :--- | :--- | :---: |
| **Multi Dataset Regression** | `Logistic Regression`, `Linear`, `Logistic`, `ElasticNet` | EDA, L1/L2 , Gradient Descent, OLS, IQR Outlier Removal, One-Hot Encoding | Achieved 96.0% accuracy on Penguins classification; successfully prevented overfitting via L1/L2 penalties | [🔗 Code](./01-supervised-learning/multi-dataset-regression) |

### 2. Deep Learning: Computer Vision & Vision Models
| Project / Domain | Implemented Algorithms | Key Tech / Methods | Summary of Results | View |
| :--- | :--- | :--- | :--- | :---: |
| **PyTorch Neural Networks to VGG-13** | `Custom NN`, `CNN`, `VGG-13` | PyTorch from scratch, Batch Norm, Early Stopping, LR Scheduling, EMNIST (36 classes) | Achieved 92.0% accuracy on EMNIST test set; demonstrated trade-offs of adapting VGG-13 to $28 \times 28$ image constraints | [🔗 Code](./02-deep-learning/pytorch-neural-networks-to-vgg13) |
| **ResNet-18 vs. VGG-16 Ablation & Vanishing Gradients** | `ResNet-18`, `VGG-16 (Ver. C)`, `VGG-Deep` | Residual Skip Connections, He vs. Xavier Init, SGD vs. Adam, L2 Gradient Norm Logging, 64x64 RGB Dataset (3 classes) | ResNet-18 achieved 98.75% accuracy (vs. VGG-16 at 92.37%); experimentally proved severe vanishing gradients in deep VGG layers via L2 norm tracking | [🔗 Code](./02-deep-learning/cnn-resnet18-vs-vgg16-ablation) |
| **Air Quality Time-Series Forecasting** | `Stacked 3-Layer LSTM`, `Unidirectional LSTM` | AirQualityUCI Dataset, Missing Value Interpolation (-200 Spikes), BPTT, Epoch & LR Sensitivity Optimization | Achieved R² of 0.80 and MAE of 0.248; demonstrated unidirectional LSTMs outperform bidirectional setups for forward hourly regression | [🔗 Code](./02-deep-learning/air-quality-lstm-forecasting) |

### 3. Reinforcement Learning
| Project / Domain | Implemented Algorithms | Key Tech / Methods | Summary of Results | View |
| :--- | :--- | :--- | :--- | :---: |
| **MDP to n-step Double Q-learning** | `Custom MDP (Gymnasium)`, `SARSA (On-Policy)`, `n-step Double Q-Learning` | Temporal-Difference Control, Safety in AI Protocols, $\epsilon$-decay Tuning, Two-Q-table Bias Mitigation | SARSA achieved superior stability and faster convergence compared to n-step Double Q-learning ($n=3$); implemented boundary safety validation without external RL solvers | [🔗 Code](./03-reinforcement-learning/mdp-to-nstep-double-q-learning) |
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
│   ├── pytorch-neural-networks-to-vgg13
│   │   ├── 01_binary_classification_nn.ipynb
│   │   ├── 02_binary_classification_nn_optimization.ipynb
│   │   ├── 03_emnist_cnn_classification.ipynb
│   │   └── 04_emnist_vgg13_adaptation.ipynb
│   ├── cnn-resnet18-vs-vgg16-ablation/
│   │   ├── 01_resnet18_vs_vgg16_ablation_study.ipynb
│   │   └── 02_vanishing_gradient_l2_norm_verification.ipynb
│   └── air-quality-lstm-forecasting
│       └── air_quality_lstm_time_series_forecasting.ipynb
├──03-reinforcement-learning
│   └── mdp-to-nstep-double-q-learning
│       ├── 01_environment_definition_and_safety_test.ipynb
│       ├── 02_sarsa_tabular_implementation.ipynb
│       └── 03_nstep_double_q_learning_analysis.ipynb
