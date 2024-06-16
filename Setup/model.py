import torch
from torch import nn
from torchvision import models


def setup_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    '''
    Sets up a specified model with pretrained weights, 
    modifies the final layer to match the number of output classes, 
    and moves the model to the specified device.

    Args:
    - model_name (str): The name of the model to setup. 
                        Supported models include 'efficientnet_b0', 'efficientnet_b1', 'resnet50', 'vgg16', and 'inception_v3'.
    - num_classes (int): The number of output classes for the final layer. This should match the number of Classes in your dataset.
    - device (torch.device): The device on which to run the model (e.g., 'cpu' or 'cuda').

    Returns:
    - nn.Module: The modified model ready for training or evaluation.

    Raises:
    - ValueError: If an unsupported model name is provided.

    Specific Notes:
    - For EfficientNet models (b0, b1), the classifier's final layer is replaced.
    - For ResNet50, the fully connected (fc) layer is replaced.
    - For VGG16, the classifier's final layer is replaced.
    - For Inception_v3, both the main classifier and auxiliary classifier (if aux_logits is True) are replaced.
    '''

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True).to(device)
        num_features = model.classifier[1].in_features                              # extracts the 'number of Input features' from the classifier's final layer
        model.classifier[1] = nn.Linear(in_features=num_features, 
                                        out_features=num_classes)                   # replaces the final linear layer with a new one,
                                                                                    # that has the Number of Output Classes = Number of Bird Species in the Dataset
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(pretrained=True).to(device)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_features, 
                                        out_features=num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True).to(device)
        num_features = model.fc.in_features
        model.fc = nn.Linear(in_features=num_features, 
                             out_features=num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True).to(device)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_features, 
                                        out_features=num_classes)

    elif model_name == "inception_v3":
        model = models.inception_v3(pretrained=True).to(device)
        num_features = model.fc.in_features
        model.fc = nn.Linear(in_features=num_features, 
                             out_features=num_classes)
        
        if model.aux_logits:                                                    # it is a boolean attribute that indicates if the model includes the auxiliary classifier
            num_features_aux = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(in_features=num_features_aux,
                                           out_features=num_classes)

    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    return model
