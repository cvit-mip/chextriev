from glob import glob
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    
    def __init__(self, config, transform=None, split='val', to_gen=-1) -> None:
        self.config = config
        split = 'validate' if 'val' in split else split
        self.split = split

        self.image_files = sorted(glob(config['data_dir'] + f'{split}/' + "*_image.npy"))
        self.is_jpg = False if len(self.image_files) > 0 else True
        self.image_files = sorted(glob(config['data_dir'] + "**/*.jpg", recursive=True)) if self.is_jpg else self.image_files
        
        if self.is_jpg:
            raise NotImplementedError('JPG not implemented yet')
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                # transforms.Normalize((0.5,), (0.5,))
            ]) if transform is None else transform

            df1 = pd.read_csv('data/mimic_cxr_jpg/mimic-cxr-2.0.0-chexpert.csv').fillna(0).replace(-1, 0)
            df2 = pd.read_csv('data/mimic_cxr_jpg/mimic-cxr-2.0.0-split.csv')
            self.df = pd.merge(df1, df2, on=['subject_id', 'study_id'])

            set_of_images = set(self.df[self.df["split"] == split]["dicom_id"].to_list())
            self.image_files = [img_file for img_file in tqdm(self.image_files, desc=split) if img_file.split("/")[-1].split(".")[0] in set_of_images]
        
        else:
            self.transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]) if transform is None else transform

        self.to_gen = to_gen if to_gen > 0 else len(self.image_files)
    
    def __len__(self):
        return self.to_gen
    
    def __getitem__(self, idx):

        try:
            # random index selection
            # idx = torch.randint(0, len(self.image_files), (1,)).item() if self.to_gen != len(self.image_files) else idx
            
            # read image
            if self.is_jpg:
                raise NotImplementedError('JPG not implemented yet')
                img = Image.open(self.image_files[idx])
                img = self.transform(img)
            
                # extract labels
                img_id = self.image_files[idx].split("/")[-1].split(".")[0]
                labels = self.df[self.df['dicom_id'] == img_id].values.tolist()[0][2:-2]
                labels = torch.tensor(labels, dtype=torch.float32)
            else:
                img = np.load(self.image_files[idx])
                img = torch.tensor(img, dtype=torch.float32)
                img = self.transform(img)

                # extract label
                pid = self.image_files[idx].split('/')[-1].split('_')[0]
                sid = self.image_files[idx].split('/')[-1].split('_')[1]
                labels = np.load(f'{self.config["data_dir"]}/{self.split}/{pid}_{sid}_labels.npy')
                labels = torch.tensor(labels, dtype=torch.float32)

        except:
            raise ValueError(f'Error in reading {self.image_files[idx]}')
            return self.__getitem__(torch.randint(0, len(self.image_files), (1,)).item())

        return img, labels
    
    def collate_fn(self, batch):
        img, labels = list(zip(*batch))
        img = torch.stack(img)
        labels = torch.stack(labels)
        return img, labels


# class CustomDataset(Dataset):
#     def __init__(self, config: dict, to_gen: int=1000):

#         self.samples = torch.randint(0, 256, (to_gen, 3, 224, 224))
#         self.labels  = torch.randint(0, config['num_classes'], (to_gen,))
#         self.to_gen  = to_gen
    
#     def __len__(self):
#         return self.to_gen
    
#     def __getitem__(self, idx):
#         return self.samples[idx].float(), self.labels[idx].long()

