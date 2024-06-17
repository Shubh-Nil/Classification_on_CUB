import torch
from torch import nn, optim

from model import setup_model

def train_model(data_loaders: dict, epochs: int, batch_size: int, learning_rate: float, num_classes: int, model_name: str, device: torch.device):
    '''

    '''
    # Define the Model
    model = setup_model(model_name = model_name,
                        num_classes = num_classes,
                        device = device)

    # Define the Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params = model.parameters(),
                          lr = learning_rate)
    
    for epoch in range(epochs):
        
