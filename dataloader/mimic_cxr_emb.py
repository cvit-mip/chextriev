from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


# class CustomDataset(Dataset):
#     def __init__(self, config: dict, to_gen: int=1000):
#         self.to_gen  = to_gen
    
#     def __len__(self):
#         return self.to_gen
    
#     def __getitem__(self, idx):
#         sample = torch.rand(512)
#         label  = torch.randint(0, 14, (14,)).float()
#         return sample, label


class CustomDataset(Dataset):
    def __init__(self, config: dict, split: str='train', to_gen: int=1000):

        with open(f'data/mimic_cxr_jpg/{split}_files.txt', 'r') as f:
            self.all_files = f.read().splitlines()
        # self.all_files = [f"{config['data_dir']}/{f}" for f in self.all_files]
        # self.all_files = self.all_files[:to_gen]

        # self.emb = [torch.load(f) for f in tqdm(self.all_files, desc=f'Loading {split} embeddings')]
        # self.label = [torch.load(f.replace('.pt', '_labels.pt')) for f in tqdm(self.all_files, desc=f'Loading {split} labels')]

        self.to_gen  = to_gen if to_gen > 0 else len(self.all_files)
    
    def __len__(self):
        return self.to_gen
    
    def __getitem__(self, idx):
        
        # emb = torch.tensor(self.emb[idx])
        # label = self.label[idx]
        emb = torch.tensor(torch.load(self.all_files[idx]))
        label = torch.load(self.all_files[idx].replace('.pt', '_labels.pt'))

        return emb, label.float()
    
    def collate_fn(self, batch):
        emb, label = list(zip(*batch))
        emb = torch.stack(emb)
        label = torch.stack(label)
        return emb, label
