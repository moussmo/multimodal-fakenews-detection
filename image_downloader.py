import os
import shutil
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

tsv_types = ['train', 'validate', 'test_public']

images_folder_path = os.path.join('data', 'images')
already_downloaded_images = []

if not os.path.exists(images_folder_path):
    os.makedirs(images_folder_path)
else : 
    already_downloaded_images = [x.strip('.jpg') for x in os.listdir(images_folder_path)]
    
for type in tsv_types : 
    print('Starting {} images download'.format(type))

    df = pd.read_csv(os.path.join('data', 'multimodal_{}.tsv'.format(type)), sep="\t")
    df = df.replace(np.nan, '', regex=True)
    df.fillna('', inplace=True)

    pbar = tqdm(initial= len(already_downloaded_images), total=len(df))

    df = df.loc[~df.id.isin(already_downloaded_images)]

    for index, row in df.iterrows():
        if row["hasImage"] == True and row["image_url"] != "" and row["image_url"] != "nan":
            image_url = row["image_url"]
            image_id = row['id']
            try : 
                response = requests.get(image_url, stream=True, verify=False)
                with open(os.path.join(images_folder_path, image_id + ".jpg"), 'wb') as image:
                    shutil.copyfileobj(response.raw, image)    
            except :
                print('image {} with url {} not downloaded'.format(image_id, image_url))
        pbar.update(1)
    print("{} : done".format(type))

            