import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from keras.datasets import mnist
from keras.models import Model
from keras.utils import to_categorical
import os

# Create figures directory if not already there
os.makedirs('figures', exist_ok=True)

def visualize_model_performance_and_features(model, model_name):
    """Generates AUC-ROC, Precision-Recall curves, and visualizes features of the best model."""
    
    # Load MNIST data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize and preprocess the data
    test_images = test_images.astype('float32') / 255
    test_labels = to_categorical(test_labels)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    # Get predictions
    test_predictions = model.predict(test_images)
    test_labels_cat = test_labels.argmax(axis=1)  # Convert one-hot to class indices
    predicted_labels = test_predictions.argmax(axis=1)  # Predicted class labels

    # AUC-ROC Curve
    fpr, tpr, _ = roc_curve(test_labels.ravel(), test_predictions.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} AUC-ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'figures/{model_name}_AUC_ROC.png')
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(test_labels.ravel(), test_predictions.ravel())
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.savefig(f'figures/{model_name}_Precision_Recall.png')
    plt.close()

    # Visualize First and Second Layer Filters
    conv_layers = [layer for layer in model.layers if 'conv' in layer.name]  # Get conv layers only
    
    if len(conv_layers) < 2:
        print(f"{model_name} has less than two convolutional layers, cannot visualize filters.")
        return

    # Visualize first layer filters (conv2d_1)
    try:
        filters, biases = conv_layers[0].get_weights()  # First conv layer
        print(f"Visualizing filters for layer: {conv_layers[0].name}")

        # Normalize filter values to 0–1 for visualization
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)

        # Visualize all the filters in the first layer
        n_filters = filters.shape[3]  # Number of output filters
        n_columns = 8  # Number of filters to display per row
        n_rows = n_filters // n_columns + 1

        plt.figure(figsize=(10, 10))
        for j in range(n_filters):
            ax = plt.subplot(n_rows, n_columns, j + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(filters[:, :, 0, j], cmap='gray')  # Visualize the first channel of each filter
        plt.suptitle(f'{model_name} - Filters of First Layer: {conv_layers[0].name}')
        plt.savefig(f'figures/{model_name}_First_Layer_Filters.png')
        plt.close()

    except Exception as e:
        print(f"Error visualizing filters for layer {conv_layers[0].name}: {e}")

    
    print(f"AUC-ROC, Precision-Recall curves, filters, and feature maps saved for {model_name}.")

def visualize_feature_maps(model, model_name):
    """
    Visualizes the feature maps for the first and second convolutional layers in the model and saves them.

    Args:
    model: The CNN model.
    model_name: Name of the model.

    Returns:
    Saves the feature maps for the first two convolutional layers.
    """
    # Load MNIST data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize and preprocess the data
    test_images = test_images.astype('float32') / 255
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    
    # Pick the first test image for visualization
    test_image = test_images[:1]  # Select the first image
    
    # Get the first two convolutional layers
    conv_layers = [layer for layer in model.layers if 'conv' in layer.name]
    if len(conv_layers) < 2:
        print(f"{model_name} does not have enough convolutional layers to visualize.")
        return

    # Loop over the first two convolutional layers
    for idx in range(2):
        # Create a new model that outputs the feature maps from the conv layer
        feature_map_model = Model(inputs=model.inputs, outputs=conv_layers[idx].output)
        
        # Predict the feature maps for the first image
        feature_maps = feature_map_model.predict(test_image)
        
        # Plot the feature maps
        n_features = feature_maps.shape[-1]  # Number of filters in the conv layer
        n_columns = 8  # Set grid columns
        n_rows = n_features // n_columns + (1 if n_features % n_columns != 0 else 0)

        plt.figure(figsize=(15, 15))
        for i in range(n_features):
            ax = plt.subplot(n_rows, n_columns, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0, :, :, i], cmap='gray')  # Display feature map
        plt.suptitle(f'{model_name} - Feature Maps from Conv Layer {idx + 1}')

        # Save the figure
        plt.savefig(f'figures/{model_name}_Feature_Maps_Layer_{idx + 1}.png')
        plt.close()

    print(f"Feature maps from the first two convolutional layers of {model_name} saved.")