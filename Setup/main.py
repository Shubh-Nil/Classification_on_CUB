import os
import subprocess
import torch
from tqdm.auto import tqdm

import config
from data_preprocessing import *
from model import *
from train import *


# check if the Dataset is already cloned
if not os.path.isdir("/home/shubhranil/Classification_on_CUB/Setup/CUB-200-2011-dataset"):
  try:
    # clone the CUB dataset from the following repository, if not available download from other sources.
    subprocess.run(["git", "clone", "https://github.com/cyizhuo/CUB-200-2011-dataset.git"], check=True)
    print("'CUB-200-2011-dataset' cloned successfully")

  except subprocess.CalledProcessError as e:
    print(f"Error occurred while cloning the repository: {e}")

else:
  print("CUB-200-2011-dataset already exists")


# Setup train and test datapath
base_dir = '/home/shubhranil/Classification_on_CUB/Setup/CUB-200-2011-dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')


# Setup device-agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device used is {device}")


# HYPERPARAMETERS
EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE
NUM_WORKERS = config.NUM_WORKERS
NUM_CLASSES = config.NUM_CLASSES
MODEL_NAME = config.MODEL_NAME
  

# Setup Dataloaders
dataloaders = create_dataloaders(train_directory = train_dir,
                                 test_directory = test_dir,
                                 model_name = MODEL_NAME,
                                 batch_size = BATCH_SIZE,
                                 num_workers = NUM_WORKERS
                                 )


# Setup Model, Loss function and Optimizer
model = setup_model(model_name = MODEL_NAME,
                    num_classes = NUM_CLASSES,
                    device = device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params = model.parameters(),
                      lr = LEARNING_RATE)


# Define Accuracy function
def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
  '''
  Out of 100 examples, what percentage does our model get right
  '''
  correct = torch.eq(y_true, y_pred).sum().item()     # computes element-wise equality
  acc = (correct/len(y_pred)) * 100
  return acc


# Setup Training and Evaluation loop
for epoch in tqdm(range(EPOCHS)):
  if epoch % 10 == 0:
    print(f"Epoch: {epoch}")

  train_step(model = model,
             dataloader = dataloaders['train'],
             loss_fn = criterion,
             accuracy_fn = accuracy_fn,
             optimizer = optimizer,
             device = device,
             epoch = epoch)
  
  test_step(model = model,
            dataloader = dataloaders['test'],
            loss_fn = criterion,
            accuracy_fn = accuracy_fn,
            device = device,
            epoch = epoch)
  
  print("\n")
