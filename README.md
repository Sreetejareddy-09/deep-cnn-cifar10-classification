
# CIFAR-10 Image Classification using Convolutional Neural Networks (CNN) - Keras

This project focuses on building and training different CNN architectures using the CIFAR-10 dataset in Keras. The experiments are designed to explore various deep learning techniques such as basic CNNs, data augmentation, and batch normalization to improve model performance and generalization.

## Dataset

The CIFAR-10 dataset is used for this project. It consists of 60,000 color images in 10 classes, with each image being 32x32 pixels. The data was loaded using keras.datasets.cifar10. It was then split into:

- Training set
- Validation set (20% of the training data)
- Test set

All image pixel values were normalized to the range [0, 1] by dividing by 255. The labels were one-hot encoded using to_categorical().

---

## Model 1: Basic CNN

A simple CNN was created with the following architecture:
- *2 Convolutional blocks*:
  - Block 1: Two Conv2D layers with 32 filters (3x3), followed by MaxPooling2D
  - Block 2: Two Conv2D layers with 64 filters (3x3), followed by MaxPooling2D
- *Flatten layer*
- *Fully Connected (FC) layer* with 512 nodes
- *Output layer* with 10 nodes (softmax)

### Training
- Optimizer: Adam (learning rate = 0.001)
- Loss: Categorical Crossentropy
- Epochs: 50
- Batch size: 32
- Best model saved using ModelCheckpoint based on lowest validation loss
- Plotted training and validation loss across epochs

### Evaluation
- Used evaluate() on test data to report accuracy and loss

---

## Model 2: CNN with Data Augmentation

The same CNN architecture was used, but this time with data augmentation using Keras’ ImageDataGenerator. The following augmentations were applied:
- Rotation (up to 10 degrees)
- Width and height shift (range: 0.1)
- Horizontal flip

### Training
- Same optimizer and loss
- Used fit() with data generator
- 50 epochs, batch size 32
- Best model saved using ModelCheckpoint
- Plotted training and validation loss

### Evaluation
- Evaluated model performance on test and validation sets
- Observed better generalization and reduced overfitting compared to Model 1

---

## Model 3: CNN with Batch Normalization

In this model, Batch Normalization layers were added:
- After each Conv2D and Dense layer (before ReLU activation)
- Bias was removed from Conv2D and Dense layers where batch norm was applied

### Training
- Optimizer: Adam (learning rate = 0.01)
- Loss: Categorical Crossentropy
- Batch size: 64
- Trained for 50 epochs
- Best model saved using ModelCheckpoint
- Plotted training and validation loss

### Evaluation
- Measured performance using evaluate() function
- Compared with Models 1 and 2 for analysis

---

## Observations

- *Model 1*: Simple CNN showed signs of overfitting as the validation loss started increasing after a point.
- *Model 2*: Data augmentation helped improve generalization and reduced overfitting. Validation loss was more stable.
- *Model 3*: Batch normalization helped accelerate training and stabilize learning. However, a higher learning rate required careful tuning.

---

## Conclusion

This project demonstrates how architectural choices and training techniques such as data augmentation and batch normalization can impact the performance and generalization of CNN models. Training visualization and evaluation metrics help in diagnosing overfitting or underfitting issues and improving model design.
