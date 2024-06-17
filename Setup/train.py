import torch
from torch import nn, optim


def train_step(model: nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: nn.Module, 
               accuracy_fn,
               optimizer: optim.Optimizer, 
               device: torch.device,
               epoch: int) -> None:
    '''
    Performs a training with model trying to learn on dataloader.
    '''
    ### Training 

    # put model into training mode
    model.train()
    loss_epoch, acc_epoch = 0, 0
    # Add a loop, to loop through training batches
    for batch, (images, labels) in enumerate(dataloader):       # images -> X, contain a 'batch of images'
                                                                # labels -> y, contain a 'batch of target labels'

        # put data on target device
        images, labels = images.to(device), labels.to(device)

        # 1. Forward pass
        logits = model(images)

        # 2. Calculate the loss
        loss_batch = loss_fn(logits, labels)
        loss_epoch += loss_batch
        # calculate the accuracy (per batch)
        acc_epoch += accuracy_fn(y_true = labels,
                                 y_pred = logits.argmax(dim=1))
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        # 4. Loss backward
        loss_batch.backward()
        # 5. Optimizer step
        optimizer.step()

    # Calculate the "train loss average per batch" / "train loss per epoch"
    loss_epoch /= len(dataloader)
    # calculate the "train acc average per batch" / "train acc per epoch"
    acc_epoch /= len(dataloader)

    if epoch % 10 == 0:
        print(f"Train loss: {loss_epoch:.5f} | Train acc: {acc_epoch:.2f}%")


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device,
              epoch: int) -> None:
    """
    Performs a testing loop step on model going over data_loader.
    """

    ### Testing

    # Put the model into evaluation mode
    model.eval()

    loss_epoch, acc_epoch = 0, 0
    with torch.inference_mode():
        for images, labels in dataloader:

            # Put data on target device
            images, labels = images.to(device), labels.to(device)

            # 1. Forward pass
            logits = model(images)

            # 2. Calculate the loss (accumulatively)
            loss_epoch += loss_fn(logits, labels)
            # Calculate the accuracy
            acc_epoch += accuracy_fn(y_true = labels,
                                     y_pred = logits.argmax(dim=1))

            # Calculate the "test loss average per batch" / "test loss per epoch"
            loss_epoch /= len(dataloader)
            # calculate the "test acc average per batch" / "test acc per epoch"
            acc_epoch /= len(dataloader)

    if epoch % 10 == 0:
        print(f"Test loss: {loss_epoch:.5f} | Test acc: {acc_epoch:.2f}%")
