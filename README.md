Convolutional Neural Network for Fashion-MNIST Classification

TensorFlow / Keras – Image Classification

Overview

This project implements and evaluates a Convolutional Neural Network (CNN) for image classification using the Fashion-MNIST dataset, a widely adopted benchmark for computer vision tasks.
The objective is to design, train, and optimize a deep learning model capable of accurately classifying grayscale fashion images into 10 distinct clothing categories.

The project covers the full deep learning workflow: data exploration, preprocessing, model design, training, regularization, evaluation, and result visualization.

Dataset

Fashion-MNIST consists of:

60,000 training images

10,000 test images

Image size: 28 × 28 grayscale

10 balanced classes:

T-shirt/top, Trouser, Pullover, Dress, Coat

Sandal, Shirt, Sneaker, Bag, Ankle boot

Each image is represented as a flattened vector of pixel intensities (0–255), later reshaped for CNN input.

Project Objectives

Explore and analyze the Fashion-MNIST dataset

Build a CNN model using TensorFlow/Keras

Evaluate model performance using accuracy and classification metrics

Identify overfitting through validation curves

Improve generalization using Dropout regularization

Visualize correct and incorrect model predictions

Data Preprocessing

Normalization of pixel values to 
[
0
,
1
]
[0,1]

Reshaping input data to (28, 28, 1) for CNN compatibility

One-hot encoding of class labels

Train/validation split (80% / 20%)

Class distributions were verified to be balanced across training, validation, and test sets.

Model Architecture

The CNN is built using a Sequential architecture with increasing feature depth:

Baseline CNN

Convolutional layers: 32 → 64 → 128 filters

Kernel size: 3 × 3

Activation: ReLU

MaxPooling layers for spatial downsampling

Fully connected layer (128 units)

Output layer with Softmax activation

Optimizer & Loss

Optimizer: Adam

Loss function: Categorical Cross-Entropy

Metric: Accuracy

Training & Evaluation

Batch size: 128

Epochs: 50

Validation set used for monitoring generalization

Initial Results

Test accuracy: ~91%

Validation loss increased after early epochs → indication of overfitting

Regularization & Model Improvement

To reduce overfitting, Dropout layers were added at multiple stages:

After convolutional blocks

Before fully connected layers

This significantly improved generalization.

Final Results (With Dropout)

Test accuracy: ~93%

Improved validation stability

Reduced gap between training and validation loss

Model Performance Analysis

Best classification performance observed for:

Trouser, Sandal, Bag, Ankle boot, Sneaker

Most challenging classes:

Shirt, Pullover (visual similarity with other classes)

A detailed classification report (precision, recall, F1-score) confirms class-wise performance variations.

Visualization

The project includes visual inspection of:

Correctly classified images

Incorrectly classified images

This qualitative analysis helps understand model confusion patterns and dataset ambiguity.

Technologies & Libraries

Python

TensorFlow / Keras

NumPy, Pandas

Scikit-learn

Matplotlib, Seaborn

Plotly

Key Skills Demonstrated

CNN architecture design and tuning

Image preprocessing for deep learning

Overfitting detection and regularization strategies

Model evaluation and error analysis

Visualization of training dynamics and predictions

Use Cases

Image classification benchmarks

Deep learning experimentation

Computer vision portfolio projects

Interview discussion for ML / AI roles

Conclusion

This project demonstrates how CNN depth, regularization, and validation analysis directly impact model performance.
By introducing Dropout layers, the model achieved strong generalization and improved accuracy on unseen data.

The workflow reflects real-world deep learning practices, from data exploration to model optimization.

Future Improvements

Data augmentation for robustness

Hyperparameter tuning (learning rate, kernel size)

Batch normalization

Model comparison with ResNet or MobileNet

Deployment as an inference API
