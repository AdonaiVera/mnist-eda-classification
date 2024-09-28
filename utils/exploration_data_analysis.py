import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def explore_dataset():
    # Step 1: Load the dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # 1. How many data samples are included in the dataset?
    num_train_samples = train_images.shape[0]
    num_test_samples = test_images.shape[0]
    total_samples = num_train_samples + num_test_samples
    print(f"Number of training samples: {num_train_samples}")
    print(f"Number of testing samples: {num_test_samples}")
    print(f"Total samples in the dataset: {total_samples}")

    # 2. Which problem will this dataset try to address?
    print("Problem: Predict handwritten digits between 0-9 (image classification problem).")

    # 3. What is the minimum value and the maximum value in the dataset?
    min_value = np.min(train_images)
    max_value = np.max(train_images)
    print(f"Minimum value in the dataset: {min_value}")
    print(f"Maximum value in the dataset: {max_value}")

    # 4. What is the dimension of each data sample?
    sample_shape = train_images.shape[1:]
    print(f"Each data sample has the shape: {sample_shape} (28x28 grayscale images)")

    # 5. Does the dataset have any missing information? E.g., missing features.
    missing_train = np.isnan(train_images).sum()
    missing_test = np.isnan(test_images).sum()
    print(f"Missing values in training set: {missing_train}")
    print(f"Missing values in test set: {missing_test}")
    if missing_train == 0 and missing_test == 0:
        print("No missing values in the dataset.")

    # 6. What is the label of this dataset?
    unique_labels = np.unique(train_labels)
    print(f"Labels of the dataset: {unique_labels} (Representing digits 0-9)")

    # 7. How many percent of data will you use for training, validation, and testing?
    print("We are going to use the same split from the original MNIST dataset.")
    
    # 8. What kind of data pre-processing will you use for your training dataset?
    # Preprocessing: Normalize the images to the range [0, 1]
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    print("Data preprocessing: Normalize images to range [0, 1]")

if __name__ == "__main__":
    explore_dataset()
