from glob import glob
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    
    def __init__(self, config, transform=None, split='val', to_gen=-1) -> None:
        
        self.config = config
        split = 'validate' if 'val' in split else split
        self.split = split

        self.image_files = sorted(glob(config['data_dir'] + f'{split}/' + "*_image.npy"))
        
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ]) if transform is None else transform

    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        
        # extract label filename
        pid = self.image_files[idx].split('/')[-1].split('_')[0]
        sid = self.image_files[idx].split('/')[-1].split('_')[1]

        # read image and label
        img = np.load(self.image_files[idx])
        labels = np.load(f'{self.config["data_dir"]}/{self.split}/{pid}_{sid}_labels.npy')
        
        # normalize image
        img = torch.tensor(img, dtype=torch.float32)
        img = self.transform(img)

        labels = torch.tensor(labels, dtype=torch.float32)

        return img, labels
