import os
import shutil
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

tsv_types = ['train', 'validate', 'test_public']

images_folder_path = os.path.join('data', 'images')
if not os.path.exists(images_folder_path):
    os.makedirs(images_folder_path)

undownloaded_images = []

for type in tsv_types : 
    df = pd.read_csv(os.path.join('data', 'multimodal_{}.tsv'.format(type)), sep="\t")
    df = df.replace(np.nan, '', regex=True)
    df.fillna('', inplace=True)

    pbar = tqdm(total=len(df))

    for index, row in df.iterrows():
        if row["hasImage"] == True and row["image_url"] != "" and row["image_url"] != "nan":
            image_url = row["image_url"]
            image_id = row['id']
            try : 
                response = requests.get(image_url, stream=True, verify=False)
                with open(os.path.join(images_folder_path, image_id + ".jpg"), 'wb') as image:
                    shutil.copyfileobj(response.raw, image)    
                pbar.update(1)
            except :
                print('image {} with url {} not downloaded'.format(image_id, image_url))
                undownloaded_images.append(image_id)
    print("{} : done".format(type))

with open(os.path.join('data', 'undownloaded_images.txt'), 'wb') as undownloaded_images_file : 
    undownloaded_images.write(undownloaded_images.join('\n'))