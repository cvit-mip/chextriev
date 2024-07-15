import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

T = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    lambda x: x*225,
    lambda x: x.squeeze(0),
])

print("Loading dataset")
df = pd.read_json('/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/mimic_coco_filtered.json')
temp_df = pd.read_csv('data/mimic_cxr_jpg/mimic-cxr-2.0.0-final.csv')
temp_df.rename(columns={'dicom_id': 'image_id'}, inplace=True)
df = df.merge(temp_df, on='image_id', how='left')
print("Dataset loaded")


class DumpDataset(Dataset):
    def __init__(self, df=df, transform=T, split='val'):
        self.split = 'validate' if 'val' in split else split
        self.df = df[df['split'] == self.split]
        self.img_loc_prefix = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
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
            sub_anatomy = self.transform(sub_anatomy)
            sub_anatomies.append(sub_anatomy)
            sub_anatomy_name.append(annotation['id'].split('_')[-1])
            sub_anatomy_labels.append(annotation['attributes'])
            
        img = self.transform(img)
        images = torch.stack([img]+sub_anatomies)
        sub_anatomy_labels = torch.tensor(sub_anatomy_labels).float()
        global_label = (torch.sum(sub_anatomy_labels, dim=0) > 0).float()
        nine_class_labels = torch.cat((global_label.unsqueeze(0), sub_anatomy_labels), dim=0)

        fourteen_class_labels = torch.from_numpy(sample_data[self.df.columns[-15:-1]].to_numpy().astype(np.float32))
        
        return {
            'id': idx,
            'images': images,
            'y': nine_class_labels,
            'anatomy_name': sub_anatomy_name,
            'y_14': fourteen_class_labels,
        }

num_nodes = 18

for split in ['val', 'test', 'train']:
    dump_dataset = DumpDataset(df, transform=T, split=split)
    dump_dataloader = DataLoader(dump_dataset, batch_size=16, shuffle=False, num_workers=32)

    save_file_data = f'/scratch/arihanth.srikar/{split}_data.bin'
    arr = np.memmap(save_file_data, dtype=np.uint8, mode='w+', shape=(len(dump_dataset), num_nodes+1, 224, 224))

    save_file_labels_9 = f'/scratch/arihanth.srikar/{split}_nine_labels.bin'
    arr_labels_9 = np.memmap(save_file_labels_9, dtype=np.int8, mode='w+', shape=(len(dump_dataset), num_nodes+1, 9))

    save_file_labels_14 = f'/scratch/arihanth.srikar/{split}_fourteen_labels.bin'
    arr_labels_14 = np.memmap(save_file_labels_14, dtype=np.int8, mode='w+', shape=(len(dump_dataset), 14))

    N = 0
    for _, batch in enumerate(tqdm(dump_dataloader, desc=f'Dumping {split} data')):
        B = batch['images'].shape[0]
        arr[N:N+B] = batch['images'].numpy().astype(np.uint8)
        arr_labels_9[N:N+B] = batch['y'].numpy().astype(np.int8)
        arr_labels_14[N:N+B] = batch['y_14'].numpy().astype(np.int8)
        N += B

    arr.flush()
    arr_labels_9.flush()
    arr_labels_14.flush()

    del arr
    del arr_labels_9
    del arr_labels_14
