# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
import torch
from torch import nn, optim, autocast
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm
import torchvision

from pathlib import Path
from typing import Callable

from tqdm.auto import tqdm
import copy

from timeit import default_timer as timer

import matplotlib.pyplot as plt

## Accuracy Function
def accuracy_fn(y_true, y_pred):
    """
    Calculates Accuracy Between Truth Labels and Predictions
    
    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    ## Variables
    correct, total = 0, 0
    
    ## Predictions
    predictions = y_pred.argmax(dim=1)
            
    ## Calculate Accuracy         
    correct += torch.eq(predictions, y_true).sum().item()
    total += y_true.size(0)
    return (correct / total) * 100

## Training Function to train the model
def train_model(model: nn.Module,
             optimizer: torch.optim,
             loss_fn: nn.Module,
             dataloader: DataLoader,
             accuracy_fn,
             device: torch.device = "cpu"):
    """
    Trains a PyTorch model for one epoch over the given dataloader.

    Performs a forward pass, computes loss and accuracy, backpropagates
    gradients, and updates model parameters in place using the optimizer.

    Args:
        model (nn.Module): The neural network to train.
        optimizer (torch.optim.Optimizer): Optimizer used to update model weights.
        loss_fn (nn.Module): Loss function used to compute training loss.
        dataloader (DataLoader): Provides batches of (inputs, targets) for training.
        accuracy_fn: Function that computes accuracy given predictions and targets.
        device (torch.device, optional): Device to run computations on. Defaults to "cpu".

    Returns:
        Accuracy and Loss of Current Epoch. The model's parameters are updated in place.
    """
    # Initialize Metrics
    train_loss, train_acc = 0, 0
    
    # Setup Model to device (If Device is set to CUDA:0 and user forgot to change the device of model)
    model.to(device)

    ## Setup GradScale
    scaler = torch.amp.GradScaler("cuda")
    
    # Train Loop 
    for batch, (X_train, y_train) in enumerate(tqdm(dataloader)):
        ## Setpu Data to Device(This step is necessary if device is on CUDA:0)
        X_train, y_train = X_train.to(device, non_blocking=True), y_train.to(device, non_blocking=True)
        
        ## Optimizer Zero Grad
        optimizer.zero_grad()
        ## Using Autocast for better and speed up performance
        with autocast("cuda", dtype=torch.float16):
            ## Forward Pass(Making Predictions/logits)
            y_pred = model(X_train)
            ## Calculating Loss
            loss = loss_fn(y_pred, y_train)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ## Accumulating Loss
        train_loss += loss.item()
        ## Accumulate Accuracy
        train_acc += accuracy_fn(y_train, y_pred)

    ## Calculate Average of Accuracy and Loss per Epoch and check what's happening
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    print(f"Train Loss = {train_loss:.5f} || Train Accuracy = {train_acc:.2f}%")
    return train_loss, train_acc

## Test Function to test the model
def test_model(model: nn.Module,
             loss_fn: nn.Module,
             dataloader: DataLoader,
             accuracy_fn,
             device: torch.device = "cpu"):
    """
    Tests a PyTorch model on the given dataloader and prints the results.

    Runs the model in evaluation mode over each batch (no backpropagation
    or weight updates), computes the loss and accuracy for each batch,
    then prints the average loss and accuracy across the entire dataset.

    Args:
        model (nn.Module): The trained neural network to test.
        loss_fn (nn.Module): Loss function used to compute test loss.
        dataloader (DataLoader): Provides batches of (inputs, targets) for testing.
        accuracy_fn: Function that computes accuracy given predictions and targets.
        device (torch.device, optional): Device to run computations on. Defaults to "cpu".

    Returns:
        Accuracy and Loss of Current Epoch. Test loss and accuracy are printed directly.
    """
    # Initialize Metrics
    test_loss, test_acc = 0, 0

    # Setup Model to device (If Device is set to CUDA:0 and user forgot to change the device of model)
    model.to(device)
    
    # Put Model in Evaluation Mode
    model.eval()
    with torch.inference_mode(): # to turn off the gradient tracking
        for X, y in dataloader:
            ## Setup data to device(Mandatory for GPU or CUDA device)
            X_test, y_test = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with autocast("cuda", dtype=torch.float16):
                ## Forward Pass (Making Predictions)
                output = model(X_test)
                ## Fetching Logits (For YOLO)
                y_pred = output[1] if isinstance(output, tuple) else output
                ## Calculating Loss
                loss = loss_fn(y_pred, y_test)
                
            ## Accumulate Loss and Accuracy
            test_loss += loss.item()
            test_acc += accuracy_fn(y_test, y_pred)

        # Adjust metrics and print out per Epoch
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
        return test_loss, test_acc

## Evaluation Function 
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn,
              device: torch.device):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    eval_loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            ## Setup data to device(Mandatory for GPU or CUDA device)
            X_test, y_test = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            with autocast("cuda",dtype= torch.float16):
                ## Forward Pass (Making Predictions)
                output = model(X_test)
                ## Fetching Logits (For YOLO)
                y_pred = output[1] if isinstance(output, tuple) else output
                ## Calculating Loss
                loss = loss_fn(y_pred, y_test)
            
            # Accumulate the loss and accuracy values per batch
            eval_loss += loss.item()
            acc += accuracy_fn(y_true=y_test, 
                                y_pred=y_pred) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        eval_loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": eval_loss,
            "model_acc": acc}


## Creating create_datasets Function 
def create_datasets(
    path: str,
    loader: Callable,
    split: str = "train",
    train_transform=None,
    eval_transform=None,
):
    """
    ### `create_datasets()`

    Creates PyTorch `ImageFolder` datasets for image classification with support for `train`, `val`, `test`, and `trainval` splits. Accepts a custom image loader and separate training and evaluation transformations. For `trainval`, returns the training and validation datasets separately while validating that their class mappings match. Raises a `ValueError` for unsupported splits.
    """
    root = Path(path)

    if split == "train":
        return torchvision.datasets.ImageFolder(
            root=root / "train",
            loader=loader,
            transform=train_transform,
        )

    elif split in ("val", "test"):
        return torchvision.datasets.ImageFolder(
            root=root / split,
            loader=loader,
            transform=eval_transform,
        )

    elif split == "trainval":
        train_ds = torchvision.datasets.ImageFolder(
            root=root / "train",
            loader=loader,
            transform=train_transform,
        )
        val_ds = torchvision.datasets.ImageFolder(
            root=root / "val",
            loader=loader,
            transform=eval_transform,
        )

        if train_ds.class_to_idx != val_ds.class_to_idx:
            raise ValueError(
                "Train and validation classes do not match."
            )

        return ConcatDataset([train_ds, val_ds])

    else:
        raise ValueError(
            f"Invalid split: {split}. "
            "Choose 'train', 'val', 'test', or 'trainval'."
        )




## Dataloader
def create_dataloaders(train_data=None, val_data=None, test_data=None, batch_size:int = 32, num_workers:int = 4):
    
    """
    ### `create_dataloaders()`

    Uses PyTorch DataLoader to Turn Dataset into Batches. create_dataloaders Needs :-
    Train_data, test_data, val_data(if you have), batch_size (Default 32) and num_workers (Default 4)
    """
    assert train_data is not None, "create_dataloaders() needs train_data"

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    train_dataloader = make_loader(train_data, shuffle=True)
    val_dataloader = make_loader(val_data, shuffle=False) if val_data is not None else None
    test_dataloader = make_loader(test_data, shuffle=False) if test_data is not None else None

    for name, dl in [("train", train_dataloader),
                     ("val", val_dataloader),
                     ("test", test_dataloader)]:
        if dl is not None:
            print(f"{name}_dataloader: {len(dl)} batches of {batch_size}")

    if not val_data and not test_data:
        return train_dataloader
    elif not test_data and val_data:
        return train_dataloader, val_dataloader
    elif test_data and not val_data:
        return train_dataloader, test_dataloade
    
    return train_dataloader, val_dataloader, test_dataloader


## Train and test Loop
def fit_model(model,
              train_model: Callable, 
              test_model: Callable, 
              train_dataloader: DataLoader, 
              test_dataloader: DataLoader, 
              loss_fn, 
              optimizer, 
              accuracy_fn, 
              epochs: int, 
              device: torch.device):
    """
    ### `fit_model()`

    Train and Evaluate model one eopch at a time. Returns History and best Epoch Result for Evaluation
    """
    ## To print total time used by Training Loop
    def print_train_time(start: float,
                         end: float,
                         device: torch.device= None):
        total_time = end - start
        print(f"Time Taken on {device} device is {total_time:.3f} seconds")

    torch.manual_seed(42)
    best_test_acc = 0
    best_test_loss = 0
    best_epoch = 0
    best_model_state = None
    
    ## Storing train and test scores records
    train_loss=[]
    test_loss=[]
    train_acc=[]
    test_acc=[]
    
    start_time= timer()
    ## Loop Per Epoch
    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch+1}\n{'-'*10}")
    
        ## Train Model
        model.train()
        ## Calling Train_model Function
        loss_train, acc_train = train_model(
            model= model,
            optimizer= optimizer,
            loss_fn= loss_fn,
            dataloader= train_dataloader,
            accuracy_fn= accuracy_fn,
            device= device
        )
        train_loss.append(loss_train)
        train_acc.append(acc_train)
        
        
        ## Calling test_model Function 
        loss_test, acc_test = test_model(
            model= model,
            loss_fn= loss_fn,
            dataloader= test_dataloader,
            accuracy_fn= accuracy_fn,
            device= device
        )
        test_loss.append(loss_test)
        test_acc.append(acc_test)
        
        # Check if current epoch is the best
        if acc_test > best_test_acc:
            best_test_acc = acc_test
            best_test_loss = loss_test
            best_epoch = epoch + 1
        
            # Save the model weights
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"🔥 New best model! Accuracy: {best_test_acc:.2f}%")

    history = {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_loss": train_loss,
        "test_loss": test_loss
    }
    end_time = timer()
    print_train_time(
            start=start_time, 
            end=end_time, 
            device=device
    )
    return history, best_model_state


## Plotting History of Model
def plot_history(history: dict):
    """
    ### `plot_history()`

    Plots History of model and returns two graphs that represents accuracy and loss of models
    """
    epochs_range = list(range(1, len(history["train_acc"]) + 1))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(epochs_range, history["train_acc"], label="Train Accuracy")
    ax1.plot(epochs_range, history["test_acc"], label="Test Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Training and Test Accuracy")
    ax1.legend()
    ax1.grid()

    ax2.plot(epochs_range, history["train_loss"], label="Train Loss")
    ax2.plot(epochs_range, history["test_loss"], label="Test Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training and Test Loss")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()
