import os
import json
import torch

def read_configuration_file(configuration_file_path):
    with open(configuration_file_path, 'r') as file:
        configuration = json.load(file)
    return configuration

def check_directories(configuration):
    os.makedirs(configuration['saved_models_dir_path'], exist_ok=True)
    data_dir_path= configuration['data_dir_path']
    try :
        for type in ['train', 'validate', 'test_public']:
            assert os.path.exists(os.path.join(data_dir_path, 'multimodal_{}.tsv'.format(type)))
    except Exception as e: 
        raise e
    
    try : 
        images_path = os.path.join(data_dir_path, 'images')
        assert os.path.isdir(images_path)
        assert len(os.listdir(images_path))
    except : 
        raise Exception("""images subdirectory in /data either doesn't exist or is empty. 
        \n Make sure to create the folder and fill it with the dataset images""")

def get_available_device():
    return ("cuda" if torch.cuda.is_available() 
              else "mps" if torch.backends.mps.is_available()
              else "cpu"
              )

def retrieve_loss_function(configuration):
    loss_function = configuration['training_parameters']['loss']
    if loss_function == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    elif loss_function == 'sigmoid' :
        return torch.nn.Sigmoid()
    else : 
        return torch.nn.CrossEntropyLoss()

def retrieve_optimizer(configuration, model_parameters):
    optimizer = configuration['training_parameters']['optimizer']['name']
    lr =  configuration['training_parameters']['optimizer']['lr']
    if optimizer=='adam':
        return torch.optim.Adam(model_parameters, lr=lr)
    elif optimizer=='sgd':
        return torch.optim.SGD(model_parameters, lr=lr)
    elif optimizer=='adagrad':
        return torch.optim.Adagrad(model_parameters, lr=lr)
    else : 
        return torch.optim.Adam(model_parameters, slr=lr)