import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, split='val', return_masked_img=False):
        self.split = 'validate' if 'val' in split else split
        self.return_masked_img = return_masked_img
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                lambda x: x.permute(1, 2, 0).unsqueeze(1).repeat(1, 3, 1, 1),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        
        # load embeddings and labels
        self.images = np.memmap(f'/scratch/arihanth.srikar/{self.split}_data.bin', dtype=np.uint8, mode='r').reshape(-1, 19, 224, 224)
        self.y_9 = np.memmap(f'/scratch/arihanth.srikar/{self.split}_nine_labels.bin', dtype=np.int8, mode='r').reshape(-1, 19, 9)
        self.y_14 = np.memmap(f'/scratch/arihanth.srikar/{self.split}_fourteen_labels.bin', dtype=np.int8, mode='r').reshape(-1, 14)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # index 0 is the global image
        # index 1-18 are the sub-anatomy images
        images = self.transform(self.images[idx])
        
        # same applies to the nine class labels
        y_9 = torch.from_numpy(self.y_9[idx]).float()
        
        # only one set of fourteen class labels
        y_14 = torch.from_numpy(self.y_14[idx]).float()

        return {
            'images': images,
            'y_9': y_9,
            'y_14': y_14,
        }
        