import sys
sys.path.append('.')

import torch
import utils.utils as utils

from tqdm import tqdm

from dataset.multimodal_dataset import MultimodalDataset
from models.multimodal.multimodal import MultimodalModel
from torch.utils.data import DataLoader

def train(model, dataloader, loss_function, optimizer):
    model.train()
    with tqdm(dataloader, unit="batch") as tepoch:
        for (X, y) in tepoch:
            X, y = list(map(lambda x: x.double().to(device), X)), y.long().to(device)

            pred = model(X)
            loss = loss_function(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            tepoch.set_postfix(loss=loss.item())

def validate(model, dataloader, loss_function):
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

    utils.check_directories(configuration)
    device = utils.get_available_device()

    batch_size = configuration['training_parameters']['batch_size']
    n_epochs = configuration['training_parameters']['number_epochs']

    multimodal_dataset_train = MultimodalDataset(configuration, mode='train')
    train_dataloader = DataLoader(multimodal_dataset_train, batch_size=batch_size, shuffle=True)

    multimodal_dataset_validate = MultimodalDataset(configuration, mode='validate')
    validate_dataloader = DataLoader(multimodal_dataset_validate, batch_size=batch_size, shuffle=True)

    multimodal_model = MultimodalModel(configuration).double().to(device)
    
    loss_function = utils.retrieve_loss_function(configuration)
    optimizer = utils.retrieve_optimizer(configuration, model_parameters=multimodal_model.parameters())

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(multimodal_model, train_dataloader, loss_function, optimizer)
        validate(multimodal_model, validate_dataloader, loss_function)

    print('Training over')
        




    



    