import sys
sys.path.append('.')

import os
import torch
import utils.utils as utils
from dataset.multimodal_dataset import MultimodalDataset
from models.multimodal.multimodal import MultimodalModel
from torch.utils.data import DataLoader

def train(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_function(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__=="__main__":
    configuration_path = r"src/config/default_configuration.json"
    configuration = utils.read_configuration_file(configuration_path)

    utils.setup_directories(configuration)
    #utils.check_data_directory(configuration)
    device = utils.get_available_device()

    batch_size = configuration['training_parameters']['batch_size']
    n_epochs = configuration['training_parameters']['number_epochs']
    loss_function = utils.retrieve_loss_function(configuration)
    optimizer = utils.retrieve_optimizer(configuration)

    multimodal_dataset_train = MultimodalDataset(configuration, mode='train')
    train_dataloader = DataLoader(multimodal_dataset_train, batch_size=batch_size)

    multimodal_dataset_validate = MultimodalDataset(configuration, mode='validate')
    validate_dataloader = DataLoader(multimodal_dataset_validate, batch_size=batch_size)

    multimodal_dataset_test = MultimodalDataset(configuration, mode='test_public')
    test_dataloader = DataLoader(multimodal_dataset_test, batch_size=batch_size)

    multimodal_model = MultimodalModel(configuration).to(device)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, multimodal_model, loss_function, optimizer)
        test(validate_dataloader, multimodal_model, loss_function)

    print('Training over')
        




    



    