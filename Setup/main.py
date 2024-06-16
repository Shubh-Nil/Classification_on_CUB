import os
import subprocess
from importlib import util

import config


# check if the Dataset is already cloned
if not os.path.isdir("CUB-200-2011-dataset"):
  try:
    # clone the CUB dataset from the following repository, if not available download from other sources.
    subprocess.run(["git", "clone", "https://github.com/cyizhuo/CUB-200-2011-dataset.git"], check=True)
    print("'CUB-200-2011-dataset' cloned successfully")

  except subprocess.CalledProcessError as e:
    print(f"Error occurred while cloning the repository: {e}")

else:
  print("CUB-200-2011-dataset already exists")


# HYPERPARAMETERS
EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE
NUM_CLASSES = config.NUM_CLASSES
MODEL_NAME = config.MODEL_NAME


# Import the necessary modules dynamically
module_names = ['data_preprocessing', MODEL_NAME, 'train', 'eval']
for module_name in module_names:
  module_path = os.path.join('models', f"{module_name}.py") if module_name == MODEL_NAME else f"{module_name}.py"

  if os.path.isfile(module_path):
    spec = util.spec_from_file_location(module_name, module_path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
  else:
    raise ImportError(f"Module {module_name}.py not found")
  

# Execute data-preprocessing
# data_preprocessing.

# Train the model
# train.train_model(EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_CLASSES, MODEL_NAME)

# Evaluate the model
# eval.evaluate_model()
