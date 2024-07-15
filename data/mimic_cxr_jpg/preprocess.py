import sys
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomDataset(Dataset):

    def __init__(self, image_files) -> None:
        self.image_files = image_files
        
        self.T = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.currupted = []

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
            
            try:
                # read image
                img = Image.open(self.image_files[idx])
                # transform image
                img = self.T(img)

                return idx, img
            
            except:
                self.currupted.append(self.image_files[idx])
                print(f'Corrupted: {self.image_files[idx]}')
                return idx, torch.zeros((1, 224, 224))


if __name__ == '__main__':

    img_dir = '/ssd_scratch/cvit/arihanth/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
    image_files = sorted(glob(img_dir + "**/*.jpg", recursive=True))

    dataset = CustomDataset(image_files)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, pin_memory=False, num_workers=8)

    for (idxs, imgs) in tqdm(dataloader): 
        for (idx, img) in zip(idxs, imgs):
            np.save(image_files[idx].replace('.jpg', '.npy'), img.numpy())
            
    print(dataset.currupted)
