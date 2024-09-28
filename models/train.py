import os
import time
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from sklearn.metrics import f1_score, roc_auc_score
import keras_tuner as kt
from models.models import build_convnet_hyper_model

def save_model_performance_plot(history, model_name, folder='figures'):
    """Saves accuracy and loss plots."""
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    filepath_acc = os.path.join(folder, f'{model_name}_accuracy.png')
    plt.savefig(filepath_acc)
    plt.close()

    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    filepath_loss = os.path.join(folder, f'{model_name}_loss.png')
    plt.savefig(filepath_loss)
    plt.close()

    print(f"{model_name} plots saved in {folder}")

def train_and_evaluate_model(model, model_name, learning_rate, folder='models'):
    """Trains and evaluates a given model with a dynamic learning rate."""
    # Compile the model with the given learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Print the model structure
    print(f"\nModel structure for {model_name}:")
    model.summary()  # This will print the model's layers and parameters

    # Load MNIST data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the data
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    # Train the model
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Training {model_name} model started at {start_time} with learning rate {learning_rate}...")
    
    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels),
                        batch_size=128)

    # Save the model
    if not os.path.exists(folder):
        os.makedirs(folder)
    model.save(os.path.join(folder, f'/weights/{model_name}_weights.keras'))
    print(f"Training complete. Model weights saved as '{model_name}_weights.keras'")

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'{model_name} Test Accuracy: {test_acc:.4f}')

    # Calculate additional metrics: F1 score and AUC
    test_predictions = model.predict(test_images)
    test_labels_cat = test_labels.argmax(axis=1)  # Convert one-hot to class indices
    predicted_labels = test_predictions.argmax(axis=1)  # Predicted class labels

    # Calculate F1 score
    f1 = f1_score(test_labels_cat, predicted_labels, average='macro')
    print(f'{model_name} F1 Score: {f1:.4f}')

    # Calculate AUC (for multi-class classification)
    auc = roc_auc_score(test_labels, test_predictions, multi_class="ovr")
    print(f'{model_name} AUC: {auc:.4f}')

    # Save performance plots
    save_model_performance_plot(history, model_name)

    return model

# Create a tuner to find the best learning rate
def tune_learning_rate():
    tuner = kt.Hyperband(
        build_convnet_hyper_model,            # Pass the model-building function
        objective='val_accuracy',       # Objective is to maximize validation accuracy
        max_epochs=10,                  # Max number of epochs to search
        factor=3,                       # Factor for decreasing the number of epochs
        directory='tuning',             # Directory to save search results
        project_name='ConvNet_LR_Tuning' # Project name for the tuning results
    )

    # Load MNIST data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Preprocess the data
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    # Perform the tuning
    tuner.search(train_images, train_labels, 
                 epochs=10, 
                 validation_data=(test_images, test_labels), 
                 batch_size=128)

    # Get the best hyperparameters (learning rate in this case)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print(f"The optimal learning rate is: {best_hps.get('learning_rate')}")
    

    return best_hps.get('learning_rate')