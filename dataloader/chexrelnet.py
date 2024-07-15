import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import torch
from torch_geometric.data import Dataset, Data
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, df, split='val'):
        super().__init__()

        self.split = 'validate' if 'val' in split else split
        self.df = df[df['split'] == self.split]
        self.img_loc_prefix = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
        self.my_transforms = transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        f_name = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/anatomy_matrix.csv'
        self.adj_mat = pd.read_csv(f_name, sep='\t').to_numpy()

    def len(self):
        return len(self.df)

    def get(self, idx):
        sample_data = self.df.iloc[idx]
        pid = str(int(sample_data['subject_id']))
        sid = str(int(sample_data['study_id']))
        image_file_location = f'{self.img_loc_prefix}/p{pid[:2]}/p{pid}/s{sid}/{sample_data["image_id"]}.jpg'
        
        img = Image.open(image_file_location)
        sub_anatomies = []
        sub_anatomy_labels = []
        sub_anatomy_name = []
        for annotation in sorted(sample_data['annotations'], key=lambda k: k['category_id']):
            x, y, w, h = annotation['bbox']
            sub_anatomy = img.crop((x, y, x+w, y+h))
            sub_anatomy = self.my_transforms(sub_anatomy)
            sub_anatomies.append(sub_anatomy)
            sub_anatomy_name.append(annotation['id'].split('_')[-1])
            sub_anatomy_labels.append(annotation['attributes'])

        sub_anatomies = torch.stack(sub_anatomies)
        sub_anatomy_labels = torch.tensor(sub_anatomy_labels).float()
        edge_list = torch.tensor(np.argwhere(self.adj_mat == 1), dtype=torch.long)
        img = self.my_transforms(img)
        complete_labels = torch.from_numpy(sample_data[self.df.columns[-15:-1]].to_numpy().astype(np.float32))

        graph_data = Data(x=sub_anatomies, edge_index=edge_list.T, y=sub_anatomy_labels)
        
        return {
            'global_feat': img,
            'label_name': sub_anatomy_name,
            'complete_labels': complete_labels,
            'graph_data': graph_data,
        }
    