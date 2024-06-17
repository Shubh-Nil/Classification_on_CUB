from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def create_dataloaders(train_directory: str, test_directory: str, model_name: str, batch_size: int, num_workers: int) -> dict:
    '''

    '''
    # define transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                 std = [0.485, 0.456, 0.406])

        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                 std = [0.485, 0.456, 0.406])
        ])
    }

    # load datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_directory, data_transforms['train']),
        'test': datasets.ImageFolder(test_directory, data_transforms['test'])
    }

    # create dataloaders
    image_dataloaders = {
        'train': DataLoader(image_datasets['train'], 
                            batch_size = batch_size, 
                            shuffle = True, 
                            num_workers = 1),
        'test': DataLoader(image_datasets['test'],
                           batch_size = batch_size,
                           shuffle = True,
                           num_workers = 1)
    }

    return image_dataloaders
