# Classification_on_CUB

This repository contains code for fine-grained classification on the CUB-200-2011 dataset using various CNN Architectures. Follow the instructions below to set up and run the project.

### Clone the Repository

```bash
git clone https://github.com/Shubh-Nil/Classification_on_CUB.git
```

### Setup the environment

```bash
conda create -n cub python==3.12
conda activate cub
```

### Install the required packages

```bash
pip install -r requirements.txt
```

### Navigate to the Setup directory, modify the hyperparameters in the config.py file as needed, and save it. Run the main.py script

```bash
cd Setup
# modify config.py
python main.py
```

### Structure of the repository

```bash
Fine_grained_classification/
│
├── CUB-200-2011-dataset/
│
├── Setup/
│   ├── notebooks/
│   │   ├── CUB_dataset.ipynb
│   │   ├── efficientnet_b0.ipynb
│   │   ├── efficientnet_b1.ipynb
│   │   ├── resnet50.ipynb
│   │   ├── vgg16.ipynb
│   │   └── inception_v3.ipynb
│   │
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── main.py
│   ├── model.py
│   └── train.py
│
├── LICENSE
├── README.md
└── requirements.txt
```


### Structure of the CUB-200-2011-dataset

```bash
CUB-200-2011-dataset/
│
├── train/
│   ├── 001.Black_footed_Albatross/
│   │   ├── image1
│   │   ├── image2
│   │   ├── ...
│   │
│   ├── 002.Laysan_Albatross/
│   ├── ...
│   └── 200.Common_Yellowthroat/
│
└── test/
    ├── 001.Black_footed_Albatross/
    ├── 002.Laysan_Albatross/
    ├── ...
    └── 200.Common_Yellowthroat/
```