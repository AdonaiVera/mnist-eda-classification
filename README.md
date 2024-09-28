# **MNIST Image Classification**

### **Student:** Adonai Vera  
### **Course:** CS 5173/6073: Deep Learning  
### **Instructor:** Dr. Jun Bai  

---

## **Project Overview**

This project involves classifying handwritten digits from the **MNIST** dataset using different deep learning architectures. The models evaluated include **Deep Neural Networks (DNN)**, **Convolutional Neural Networks (ConvNet)**, **VGG-like** networks, and **ResNet**. Each model was trained and fine-tuned to optimize accuracy, learning rate, and other hyperparameters.

---

## **Dataset**

The dataset used for this project is the **MNIST** dataset, which contains 70,000 grayscale images of handwritten digits (0-9). Each image is 28x28 pixels.

---

## **Project Steps**

1. **Exploratory Data Analysis (EDA)**:
   - Performed EDA to check the dataset properties, distribution of labels, and missing values.
   - The analysis code is available in `utils/exploration_data_analysis.py`.

2. **Data Preprocessing**:
   - Normalized pixel values (scaling them to the [0, 1] range).
   - Reshaped images for compatibility with the neural network input layers.
   - Split the dataset into training, validation, and test sets.
   - Preprocessing steps can be found in the respective functions in `train.py`.

3. **Model Training**:
   The following models were trained and fine-tuned:
   - **DNN**: A deep neural network with dense layers.
   - **ConvNet**: A custom convolutional neural network (CNN).
   - **VGG-like Network**: A simplified version of the VGG architecture.
   - **ResNet50**: A residual network model applied to MNIST with input resizing.
   - **Hyperparameter Tuning**: For ConvNet, we performed learning rate tuning using Keras Tuner.

4. **Evaluation**:
   - Metrics: **Accuracy**, **F1 Score**, and **AUC-ROC** were used to evaluate each model.
   - Plots: Model loss and accuracy plots were generated for each architecture.
   - The **best model** was identified based on these metrics and further visualizations of filters were produced.

---

## **Setup Instructions**

### **Prerequisites**

- Python 3.8 or higher.
- Required libraries listed in `requirements.txt`.

### **Installing Dependencies**

To install all necessary packages, run:

```bash
pip install -r requirements.txt
```

### **Running the Project**

To train the models, execute:

```bash
python main.py
```

This script will:
- Load and preprocess the MNIST dataset.
- Train **DNN**, **ConvNet**, **VGG-like**, and **ResNet** models.
- Evaluate the models on the test set and print performance metrics.

---

## **Testing Models**

To test the best trained model, run:

```bash
python test_model.py
```

This script will:
- Load the **fine-tuned ConvNet** model.
- Preprocess the test dataset.
- Generate predictions and output performance metrics such as **F1 Score** and **AUC**.

---

## **Project Structure**

```
├── models/
│   ├── models.py                   # Model architecture definitions (DNN, ConvNet, VGG, ResNet)
│   ├── weights/                    # Saved model weights after training
├── train.py                         # Model training script with evaluation
├── main.py                          # Main entry point to run the models
├── test_model.py                    # Model testing and evaluation script
├── utils/
│   ├── exploration_data_analysis.py # EDA and data loading scripts
│   ├── visualize_metrics.py         # Helper functions to visualize performance
├── figures/                         # Figures and performance plots generated
├── README.md                        # Project description and instructions
```

---

## **Model Details**

### **DNN**:
- A basic dense neural network applied to the MNIST dataset.
- **Performance**:
  - Test Accuracy: 98.14%
  - F1 Score: 0.9813
  - AUC: 0.9991

### **ConvNet**:
- A custom CNN with three convolutional layers and max-pooling.
- **Performance**:
  - Test Accuracy: 99.29%
  - F1 Score: 0.9931
  - AUC: 1.0000

### **VGG-like Network**:
- A simplified VGG architecture designed to classify MNIST digits.
- **Performance**:
  - Test Accuracy: 99.25%
  - F1 Score: 0.9925
  - AUC: 1.0000

### **ResNet50**:
- Residual network applied to MNIST with input resizing from 28x28 to 32x32.
- **Performance**:
  - Test Accuracy: 98.62%
  - F1 Score: 0.9861
  - AUC: 1.0000

---

## **Model Comparison**

| Model                  | Test Accuracy | F1 Score | AUC    |
|------------------------|---------------|----------|--------|
| DNN                    | 98.14%        | 0.9813   | 0.9991 |
| ConvNet                | 99.29%        | 0.9931   | 1.0000 |
| VGG-like               | 99.25%        | 0.9925   | 1.0000 |
| ResNet50               | 98.62%        | 0.9861   | 1.0000 |

---

## **Conclusion**

The **ConvNet** model with auto-tuned learning rate performed the best, achieving a test accuracy of **99.29%**, followed closely by the **VGG-like** network. The learning rate tuning was especially beneficial for improving convergence and model stability. Deeper models like **ResNet50** also performed well but did not outperform the simpler architectures for this particular dataset.