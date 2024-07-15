import numpy as np
from glob import glob
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


fourteen_class_labels = [  
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Enlarged Cardiomediastinum',
    'Fracture',
    'Lung Lesion',
    'Lung Opacity',
    'No Finding',
    'Pleural Effusion',
    'Pleural Other',
    'Pneumonia',
    'Pneumothorax',
    'Support Devices']

nine_class_labels = [
    'lung opacity', 
    'pleural effusion', 
    'atelectasis', 
    'enlarged cardiac silhouette',
    'pulmonary edema/hazy opacity', 
    'pneumothorax', 
    'consolidation', 
    'fluid overload/heart failure', 
    'pneumonia']

anatomy_names = [
    'right lung',
    'right apical zone',
    'right upper lung zone',
    'right mid lung zone',
    'right lower lung zone',
    'right hilar structures',
    'right costophrenic angle',
    'left lung',
    'left apical zone',
    'left upper lung zone',
    'left mid lung zone',
    'left lower lung zone',
    'left hilar structures',
    'left costophrenic angle',
    'mediastinum',
    'upper mediastinum',
    'cardiac silhouette',
    'trachea']


class CustomDataset(Dataset):
    def __init__(self, df, split='val', return_masked_img=False):
        self.split = 'validate' if 'val' in split else split
        self.df = df[df['split'] == self.split]
        self.return_masked_img = return_masked_img
        self.img_loc_prefix = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
        if not Path(self.img_loc_prefix).is_dir():
            self.img_loc_prefix = '/scratch/arihanth.srikar/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
        self.transform = transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        self.masked_transform = transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_data = self.df.iloc[idx]

        pid = str(int(sample_data['subject_id']))
        sid = str(int(sample_data['study_id']))
        image_file_location = f'{self.img_loc_prefix}/p{pid[:2]}/p{pid}/s{sid}/{sample_data["image_id"]}.jpg'
        img = Image.open(image_file_location)
        
        # initialise sub-anatomy masked global image
        masked_imges = []
        
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
            
            if self.return_masked_img:
                # create an image with all 1 the size of the original image
                # 0 out the pixels in the sub-anatomy
                masked_img = Image.new('L', (img.size[0], img.size[1]), 255)
                masked_img.paste(0, (x, y, x+w, y+h))
                masked_img = self.masked_transform(masked_img)
                masked_imges.append(masked_img)

        masked_imges = torch.stack(masked_imges) if self.return_masked_img else []
        sub_anatomies = torch.stack(sub_anatomies)
        sub_anatomy_labels = torch.tensor(sub_anatomy_labels).float()
        img = self.transform(img)
        complete_labels = torch.from_numpy(sample_data[self.df.columns[-15:-1]].to_numpy().astype(np.float32))
        
        return {
            'idx': idx,
            'global_feat': img,
            'node_feat': sub_anatomies,
            'y': sub_anatomy_labels,
            'anatomy_name': sub_anatomy_name,
            'complete_labels': complete_labels,
            'masked_img': masked_imges,
        }


class OcclusionDataset(Dataset):
    def __init__(self, df, split='val', occlude_anatomy=[], precise=False):
        self.split = 'validate' if 'val' in split else split
        self.df = df[df['split'] == self.split]
        self.precise = precise
        
        self.occlude_anatomy = occlude_anatomy
        assert set(self.occlude_anatomy).issubset(set(anatomy_names)) or len(self.occlude_anatomy) == 0, f'Invalid anatomy names: {self.occlude_anatomy}'
        
        self.img_loc_prefix = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
        if not Path(self.img_loc_prefix).is_dir():
            self.img_loc_prefix = '/scratch/arihanth.srikar/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
        self.transform = transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        self.masked_transform = transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.ToTensor(),
            ])
        
        self.cache_img = None
        self.cache_idx = None

    def __len__(self):
        return len(self.df)

    def load_datapoint(self, idx, my_mask=None):
        sample_data = self.df.iloc[idx]
        
        pid = str(int(sample_data['subject_id']))
        sid = str(int(sample_data['study_id']))
        image_file_location = f'{self.img_loc_prefix}/p{pid[:2]}/p{pid}/s{sid}/{sample_data["image_id"]}.jpg'
        img = Image.open(image_file_location) if self.cache_img is None or self.cache_idx != idx else self.cache_img
        img_size = torch.tensor(img.size)
        self.cache_img = img
        self.cache_idx = idx
        
        masked_img = Image.new('L', img.size, 255)
        if my_mask is None:
            if len(self.occlude_anatomy):
                for annotation in sorted(sample_data['annotations'], key=lambda k: k['category_id']):
                    a_name = annotation['id'].split('_')[-1]
                    if a_name in self.occlude_anatomy:
                        x, y, w, h = annotation['bbox']
                        masked_img.paste(0, (x, y, x+w, y+h))

                if self.precise:
                    for annotation in sorted(sample_data['annotations'], key=lambda k: k['category_id']):
                        a_name = annotation['id'].split('_')[-1]
                        if a_name in self.occlude_anatomy:
                            continue
                        for consider_aname in self.occlude_anatomy:
                            if 'lung' in consider_aname and 'lung' in a_name:
                                continue
                            x, y, w, h = annotation['bbox']
                            masked_img.paste(255, (x, y, x+w, y+h))
        else:
            assert len(my_mask.shape) == len(img.size), f'Invalid mask shape: {my_mask.shape}'
            my_mask = transforms.Resize(img.size, antialias=True)(my_mask.unsqueeze(0)).squeeze(0)
            my_mask = Image.fromarray(np.uint8(my_mask.numpy()*255))
            masked_img = my_mask

        sub_anatomies = []
        sub_anatomy_labels = []
        sub_anatomy_name = []
        sub_anatomy_masks = []
        for annotation in sorted(sample_data['annotations'], key=lambda k: k['category_id']):
            x, y, w, h = annotation['bbox']
            
            sub_anatomy = img.crop((x, y, x+w, y+h))
            sub_anatomy = self.transform(sub_anatomy)
            
            sub_anatomy_mask = masked_img.crop((x, y, x+w, y+h))
            sub_anatomy_mask = self.masked_transform(sub_anatomy_mask)

            sub_anatomies.append(sub_anatomy*sub_anatomy_mask)
            sub_anatomy_masks.append(sub_anatomy_mask)
            
            sub_anatomy_name.append(annotation['id'].split('_')[-1])
            sub_anatomy_labels.append(annotation['attributes'])

        img = self.transform(img)
        masked_img = self.masked_transform(masked_img)
        img = img*masked_img

        sub_anatomies = torch.stack(sub_anatomies)
        sub_anatomy_masks = torch.stack(sub_anatomy_masks)
        sub_anatomy_labels = torch.tensor(sub_anatomy_labels).float()
        # complete_labels = torch.from_numpy(sample_data[self.df.columns[-15:-1]].to_numpy().astype(np.float32))

        return {
            'idx': idx,
            'global_feat': img,
            'node_feat': sub_anatomies,
            'y': sub_anatomy_labels,
            'anatomy_name': sub_anatomy_name,
            'complete_labels': torch.zeros(14),
            # 'complete_labels': complete_labels,
            'masked_img': masked_img,
            'sub_anatomy_masks': sub_anatomy_masks,
            'img_size': img_size,
        }

    def __getitem__(self, idx):
        return self.load_datapoint(idx)

class DiseaseOcclusionDataset(Dataset):
    def __init__(self, df, split='val', consider_disease='original', precise=False):
        self.split = 'validate' if 'val' in split else split
        self.df = df[df['split'] == self.split]
        self.precise = precise
        
        self.consider_disease = consider_disease
        assert self.consider_disease in nine_class_labels + ["original"], f'Invalid disease name: {self.consider_disease}'
        
        self.img_loc_prefix = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
        if not Path(self.img_loc_prefix).is_dir():
            self.img_loc_prefix = '/scratch/arihanth.srikar/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
        self.transform = transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        self.masked_transform = transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_data = self.df.iloc[idx]

        pid = str(int(sample_data['subject_id']))
        sid = str(int(sample_data['study_id']))
        image_file_location = f'{self.img_loc_prefix}/p{pid[:2]}/p{pid}/s{sid}/{sample_data["image_id"]}.jpg'
        img = Image.open(image_file_location)
        
        sub_anatomy_labels = []
        sub_anatomies = []
        sub_anatomy_name = []
        
        masked_img = Image.new('L', (img.size[0], img.size[1]), 255)
        if self.consider_disease != "original":
            consider_aname = self.consider_disease
            for annotation in sorted(sample_data['annotations'], key=lambda k: k['category_id']):
                disease_idx = nine_class_labels.index(self.consider_disease)
                if annotation['attributes'][disease_idx]:
                    consider_aname = annotation['id'].split('_')[-1]
                    x, y, w, h = annotation['bbox']
                    masked_img.paste(0, (x, y, x+w, y+h))
            if self.precise:
                for annotation in sorted(sample_data['annotations'], key=lambda k: k['category_id']):
                    disease_idx = nine_class_labels.index(self.consider_disease)
                    if annotation['attributes'][disease_idx]:
                        continue
                    a_name = annotation['id'].split('_')[-1]
                    if consider_aname == 'left lung' or consider_aname == 'right lung':
                        if 'lung' in a_name:
                            continue
                    elif 'lung' in consider_aname:
                        if a_name == 'left lung' or a_name == 'right lung':
                            continue
                    x, y, w, h = annotation['bbox']
                    masked_img.paste(255, (x, y, x+w, y+h))

        for annotation in sorted(sample_data['annotations'], key=lambda k: k['category_id']):
            x, y, w, h = annotation['bbox']
            
            sub_anatomy = img.crop((x, y, x+w, y+h))
            sub_anatomy = self.transform(sub_anatomy)
            
            sub_anatomy_mask = masked_img.crop((x, y, x+w, y+h))
            sub_anatomy_mask = self.masked_transform(sub_anatomy_mask)

            sub_anatomy_name.append(annotation['id'].split('_')[-1])
            sub_anatomy_labels.append(annotation['attributes'])

            sub_anatomies.append(sub_anatomy*sub_anatomy_mask)

        img = self.transform(img)
        masked_img = self.masked_transform(masked_img)
        img = img*masked_img
        
        sub_anatomies = torch.stack(sub_anatomies)
        sub_anatomy_labels = torch.tensor(sub_anatomy_labels).float()
        complete_labels = torch.from_numpy(sample_data[self.df.columns[-15:-1]].to_numpy().astype(np.float32))
        
        return {
            'idx': idx,
            'global_feat': img,
            'node_feat': sub_anatomies,
            'y': sub_anatomy_labels,
            'anatomy_name': sub_anatomy_name,
            'complete_labels': complete_labels,
            'masked_img': masked_img,
        }


class LocalFeatures(Dataset):
    def __init__(self, df, split='val', return_masked_img=False):
        self.split = 'validate' if 'val' in split else split
        self.df = df[df['split'] == self.split]
        
        self.img_loc_prefix = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
        if not Path(self.img_loc_prefix).is_dir():
            self.img_loc_prefix = '/scratch/arihanth.srikar/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
        self.transform = transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def load_datapoint(self, idx, my_mask=None):
        sample_data = self.df.iloc[idx]
        
        pid = str(int(sample_data['subject_id']))
        sid = str(int(sample_data['study_id']))
        image_file_location = f'{self.img_loc_prefix}/p{pid[:2]}/p{pid}/s{sid}/{sample_data["image_id"]}.jpg'
        img = Image.open(image_file_location)
        img_size = torch.tensor(img.size)
        
        x_dim, y_dim = img.size
        # x_cuts = 2
        # y_cuts = 3
        x_cuts = 5
        y_cuts = 5
        w = x_dim//x_cuts+1
        h = y_dim//y_cuts+1

        sub_anatomies = []
        for y in range(0, y_dim, h):
            for x in range(0, x_dim, w):
                sub_anatomy = img.crop((x, y, x+w, y+h))
                sub_anatomy = self.transform(sub_anatomy)
                sub_anatomies.append(sub_anatomy)

        sub_anatomy_labels = []
        for annotation in sorted(sample_data['annotations'], key=lambda k: k['category_id']):
            sub_anatomy_labels.append(annotation['attributes'])

        img = self.transform(img)
        img = img

        sub_anatomies = torch.stack(sub_anatomies)
        sub_anatomy_labels = torch.tensor(sub_anatomy_labels)
        sub_anatomy_labels = (torch.sum(sub_anatomy_labels, dim=0) > 0).float()
        sub_anatomy_labels = sub_anatomy_labels.unsqueeze(0).repeat(sub_anatomies.size(0), 1)

        return {
            'idx': idx,
            'global_feat': img,
            'node_feat': sub_anatomies,
            'y': sub_anatomy_labels,
            'img_size': img_size,
            'masked_img': 'None',
        }

    def __getitem__(self, idx):
        return self.load_datapoint(idx)