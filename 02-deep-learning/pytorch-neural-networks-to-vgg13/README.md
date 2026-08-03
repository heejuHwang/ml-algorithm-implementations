# Deep Learning Architecture & Optimization: From Scratch NN to VGG-13
> Implementing fully connected Neural Networks (NN), custom Convolutional Neural Networks (CNN), and an adapted VGG-13 architecture using PyTorch without pre-built model libraries.

---

## 📌 Project Overview

This project focuses on designing, training, and optimizing deep learning architectures from scratch using the PyTorch framework. Strictly avoiding pre-trained or pre-built model modules, the primary objective is to gain deep structural and mathematical insight into neural network behavior across tabular binary classification and multi-class image recognition tasks.

Key technical objectives include:
- **Architecture Construction:** Designing custom fully connected neural networks and convolutional networks (CNNs) by assembling foundational PyTorch building blocks (`nn.Module`, `nn.Linear`, `nn.Conv2d`).
- **Model Adaptation:** Customizing the classic **VGG-13 (Version B)** architecture to process small-scale $28 \times 28$ grayscale image inputs without losing spatial dimensionality.
- **Systematic Optimization:** Mitigating overfitting and accelerating convergence through Batch Normalization, Dropout, Early Stopping, and Learning Rate Scheduling.

---

## 🛠️ Tech Stack & Methodology

- **Framework & Libraries:** PyTorch, Scikit-learn, NumPy, Pandas, Matplotlib / Seaborn
- **Data Preprocessing & Pipeline:**
  - **Tabular Data (Binary Target):** Handled a 7-feature binary dataset by encoding categorical string features, scaling continuous features via `StandardScaler`, and applying oversampling techniques to balance class distributions.
  - **Vision Data (EMNIST):** Processed the EMNIST dataset comprising 36 distinct alphanumeric categories with 2,800 grayscale ($28 \times 28$) images per category (100,800 total samples).
- **Implemented Architectures:**
  - **Fully Connected NN:** Configured 23 input features (post-encoding) with two hidden layers utilizing **ELU activations** (selected to preserve negative values and prevent dead neurons) and a Sigmoid output neuron for binary classification.
  - **Custom CNN:** Constructed a 4-hidden-layer network for 1-channel grayscale inputs predicting 36 output classes. Utilized two convolutional layers (32 and 64 filters), Max Pooling, ReLU activations, two fully connected layers (128 units), Batch Normalization, and the Adam optimizer.
  - **Adapted VGG-13 (Version B):** Modified standard VGG-13 parameters for $28 \times 28$ resolution by reducing Max Pooling layers from five to two and incorporating custom spatial padding to prevent premature feature map collapse.

---

## 📊 Experiments & Model Performance

Models were trained and evaluated on both tabular binary classification and 36-class character recognition benchmarks, consistently surpassing target accuracy thresholds.

| Model Architecture | Task & Dataset | Key Optimizations / Hyperparameters | Test Accuracy | Evaluation Metrics |
| :--- | :--- | :--- | :---: | :--- |
| **Baseline NN** | Binary Classification (7 features) | 2 Hidden Layers (ELU), Dropout, SGD Optimizer | > 75.0% | ROC-AUC, Confusion Matrix |
| **Optimized NN** | Binary Classification (7 features) | Batch Norm, LR Scheduler, Early Stopping | > 80.0% | ROC Curve, Precision / Recall |
| **Custom CNN** | EMNIST Character (36 classes) | 2 Conv Layers (32/64 filters), Adam, Batch Norm | 92.0% | Confusion Matrix across 36 classes |
| **Adapted VGG-13** | EMNIST Character (36 classes) | Reduced Pooling (5 → 2 layers), Spatial Padding | > 85.0% | Comparative Loss & Convergence Plots |

### Key Insights
- **Model-Data Fit & Scale Trade-offs:** An unexpected empirical finding was that larger architectures do not universally guarantee superior performance on smaller-scale inputs. While VGG-13 is highly capable for complex $224 \times 224$ images, the simpler Custom CNN achieved higher efficiency and better accuracy (92.0%) on simple $28 \times 28$ EMNIST characters.
- **Impact of Learning Rate Scheduling:** Incorporating learning rate schedulers proved critical during multi-class CNN optimization, smoothing gradient updates and preventing loss divergence around local minima.

---

## 💡 Troubleshooting & Learnings

- **Spatial Collapse in Deep Convolutional Networks:** When initially applying VGG-13 to $28 \times 28$ EMNIST images, standard 5-stage pooling rapidly reduced feature map dimensions to zero. Replacing deep pooling operations with 2-stage pooling and adjusting convolutional padding successfully preserved spatial representations.
- **Combatting Overfitting in Deep Tabular NN:** The fully connected neural network exhibited training overfitting due to feature sparsity after categorical oversampling. Integrating Batch Normalization between ELU layers alongside patience-based Early Stopping significantly enhanced validation stability.