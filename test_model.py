import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

def test_model(model_path, X_test, y_test):
    """
    Function to load a trained ConvNet model and test it on the provided test dataset.

    Parameters:
    model_path (str): Path to the trained model file.
    X_test (ndarray): Preprocessed test data (features).
    y_test (ndarray): True test labels.
    """
    # Load the trained ConvNet model
    print(f"Loading the trained ConvNet model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Evaluate the model
    print("Evaluating the model on the test dataset...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Make predictions
    print("Making predictions on the test dataset...")
    test_predictions = model.predict(X_test)
    y_true = y_test.argmax(axis=1)  # Convert one-hot to class indices
    y_pred = test_predictions.argmax(axis=1)

    # Calculate F1 Score
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"F1 Score: {f1:.4f}")

    # Calculate AUC score (for multi-class classification)
    auc = roc_auc_score(y_test, test_predictions, multi_class='ovr')
    print(f"AUC Score: {auc:.4f}")

    return test_accuracy, f1, auc


def main():
    """
    Main function to load data, preprocess it, and test the fine-tuned ConvNet model.
    """
    print("Starting the ConvNet model evaluation process...\n")

    # Step 1: Load and preprocess the data
    print("Loading and preprocessing the test data...")
    
    # Load MNIST data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Preprocess the test data
    test_images = test_images.astype('float32') / 255  # Normalize the images
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)  # Reshape to match ConvNet input
    test_labels = to_categorical(test_labels)  # Convert labels to one-hot encoding

    print("Data successfully loaded and preprocessed.\n")

    # Step 2: Test the fine-tuned ConvNet model
    print("Testing the fine-tuned ConvNet Model:")
    test_model(model_path='models/weights/ConvNet_AutoTuned_LR_weights.keras', X_test=test_images, y_test=test_labels)
    print("ConvNet Model testing completed.\n")

    print("Model evaluation process completed.")

if __name__ == "__main__":
    main()
