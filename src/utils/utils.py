import os
import json

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
        raise Exception("images subdirectory in /data either doesn't exist or is empty. \
                        Make sure to create the folder and fill it with the dataset images")