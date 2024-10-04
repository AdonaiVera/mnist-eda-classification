from models.models import build_dnn_model, build_convnet_model, build_vgg_model, build_resnet_model, build_convnet_hyper_model
from models.train import train_and_evaluate_model, tune_learning_rate
from utils.visualizate_metrics import visualize_model_performance_and_features, visualize_feature_maps

# Define the default learning rate
DEFAULT_LR = 0.001

if __name__ == "__main__":
    print("Starting training with the default learning rate...")

    # Train and evaluate DNN model with default learning rate
    dnn_model = build_dnn_model()
    train_and_evaluate_model(dnn_model, "DNN_Default_LR", learning_rate=DEFAULT_LR)

    # Train and evaluate ConvNet model with default learning rate
    convnet_model = build_convnet_model()
    train_and_evaluate_model(convnet_model, "ConvNet_Default_LR", learning_rate=DEFAULT_LR)

    # Train and evaluate VGG-like model with default learning rate
    vgg_model = build_vgg_model()
    train_and_evaluate_model(vgg_model, "VGG_Default_LR", learning_rate=DEFAULT_LR)

    # Train and evaluate ResNet18 model with default learning rate
    resnet_model = build_resnet_model()
    train_and_evaluate_model(resnet_model, "ResNet50_Default_LR", learning_rate=DEFAULT_LR)

    print("\nStarting training with dynamic learning rates...")

    # Now train and evaluate each model with different learning rates
    for lr in [0.1, 0.01, 0.001]:
        print(f"\nTraining with learning rate: {lr}")
        
        # DNN with dynamic learning rate
        dnn_model = build_dnn_model()
        train_and_evaluate_model(dnn_model, f"DNN_LR_{lr}", learning_rate=lr)

        # ConvNet with dynamic learning rate
        convnet_model = build_convnet_model()
        train_and_evaluate_model(convnet_model, f"ConvNet_LR_{lr}", learning_rate=lr)

        # VGG-like model with dynamic learning rate
        vgg_model = build_vgg_model()
        train_and_evaluate_model(vgg_model, f"VGG_LR_{lr}", learning_rate=lr)

        # ResNet50 model with dynamic learning rate
        resnet_model = build_resnet_model()
        train_and_evaluate_model(resnet_model, f"ResNet50_LR_{lr}", learning_rate=lr)
        
    
    # Train the best ConvNet model with the optimal learning rate
    convnet_model = build_convnet_model()
    #train_and_evaluate_model(convnet_model, "ConvNet_AutoTuned_LR", learning_rate=tune_learning_rate())
    train_and_evaluate_model(convnet_model, "ConvNet_AutoTuned_LR", learning_rate=0.001)

    # Now visualize performance and features
    visualize_model_performance_and_features(convnet_model, "ConvNet_BestModel")
    visualize_feature_maps(convnet_model, "ConvNet_BestModel")
