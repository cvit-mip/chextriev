import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, config, split='val'):
        print(f'Using the images only dataset for {split}')
        self.config = config
        self.split = 'validate' if 'val' in split else split
        df = pd.read_csv('data/mimic_cxr_jpg/mimic-cxr-2.0.0-final.csv')
        df_new = pd.read_pickle(f"/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/scene_tabular/processed_bboxes.pkl")
        df_new.rename({'image_id': 'dicom_id'}, axis=1, inplace=True)
        df_new = pd.merge(df, df_new, on=['dicom_id'])
        df = df_new[df.columns].drop_duplicates()
        self.df = df[df['split'] == self.split]

        self.my_transforms = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(row['subject_id'])
        sid = str(row['study_id'])
        img_id = row['dicom_id']
        img_fname = f'{self.config["data_dir"]}p{pid[:2]}/p{pid}/s{sid}/{img_id}.jpg'
        
        img = Image.open(img_fname)
        img = self.my_transforms(img)
        
        label = torch.from_numpy(row[self.df.columns[2:16]].values.astype(np.float32))
        
        return img, label
    