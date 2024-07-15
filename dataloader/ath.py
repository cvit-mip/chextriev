import numpy as np
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, df, split='val', return_masked_img=False):
        self.split = 'validate' if 'val' in split else split
        self.df = df[df['split'] == self.split]
        self.return_masked_img = return_masked_img
        self.img_loc_prefix = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
        self.transform = transforms.Compose([
                transforms.Resize((256, 256), antialias=True),
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_data = self.df.iloc[idx]

        pid = str(int(sample_data['subject_id']))
        sid = str(int(sample_data['study_id']))
        image_file_location = f'{self.img_loc_prefix}/p{pid[:2]}/p{pid}/s{sid}/{sample_data["image_id"]}.jpg'
        img = Image.open(image_file_location)
        
        sub_anatomy_labels = torch.tensor([annotation['attributes'] for annotation in sorted(sample_data['annotations'], key=lambda k: k['category_id'])], dtype=torch.float)
        
        y = (torch.sum(sub_anatomy_labels, dim=0) > 0).float()
        img = self.transform(img)
        
        return {
            'x': img,
            'y': y,
            'y_9': sub_anatomy_labels,
        }
    