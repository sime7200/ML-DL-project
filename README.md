# 🤖 Machine Learning & Deep Learning Projects

This repository contains a collection of six projects that I implemented as part of my learning journey in Machine Learning and Deep Learning. Each project demonstrates a core concept or algorithm, implemented from scratch and compared with results from `scikit-learn` and `Weka`.

## 🔍 Machine Learning Projects

### 📈 1. Multivariate Linear Regression
- **Dataset**: [Automobile dataset](https://archive.ics.uci.edu/ml/datasets/automobile)
- **Goal**: Predict target variables based on multiple features. I used price as target
- **Implementation**: From-scratch multivariate linear regression.
- **Comparison**: Results validated and compared with `sklearn.linear_model.LinearRegression` and Weka `functions.LinearRegression`.

### 📊 2. K-Nearest Neighbors (KNN) Classification
- **Dataset**: [Hepatitis dataset](https://archive.ics.uci.edu/dataset/46/hepatitis)
- **Goal**: Perform classification using a distance-based voting mechanism.
- **Implementation**: Custom implementation of the KNN algorithm.
- **Comparison**: Benchmarked against `sklearn.neighbors.KNeighborsClassifier` and Weka `lazy.IBk`.

### 🧮 3. Principal Component Analysis (PCA)
- **Dataset**: [Automobile dataset](https://archive.ics.uci.edu/ml/datasets/automobile)
- **Goal**: Reduce the dimensionality of the dataset while preserving as much variance as possible.
- **Implementation**: PCA from scratch using NumPy (eigen decomposition).
- **Comparison**: Cross-checked with `sklearn.decomposition.PCA` and Weka `filters.unsupervised.attribute.PrincipalComponents`.

### 🟢 4. Single Layer Perceptron
- **Dataset**: Iris dataset.
- **Goal**: Binary classification — distinguish Iris-setosa from other species.
- **Implementation**: From scratchwith activation function.
- **Comparison**: Cross-checked with `sklearn.linear_model.Perceptron` using te same preprocessed data.


---

## 🧠 Deep Learning Projects

### 🧼 4. Image Denoising with Autoencoders
- **Datasets**: MNIST and Zalando Fashion-MNIST
- **Goal**: Train an Autoencoder to remove noise from input images.
- **Implementation**: 
  - Fully connected and convolutional autoencoders.
  - Optionally includes skip connections for performance improvement (still to add).
- **Libraries**: Implemented using Keras.

### 🎯 5. Object Detection with YOLO
- **Goal**: Detect and localize objects in images using the YOLO (You Only Look Once) architecture.
- **Implementation**: 
  - Model trained and evaluated using pre-trained YOLO weights.
  - Inference applied on custom and standard datasets.
  
