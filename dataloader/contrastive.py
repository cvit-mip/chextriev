import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, df, split='val'):
        self.split = 'validate' if 'val' in split else split
        self.df = df[df['split'] == self.split]
        self.nine_labels_list = np.array(self.df['annotations'].apply(lambda x: (np.array([y['attributes'] for y in x]).sum(axis=0) > 0).astype(int)).tolist())
        self.img_loc_prefix = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
        self.transform = transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)
    
    def get_sample(self, sample_data):
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
            
            # create an image with all 1 the size of the original image
            # 0 out the pixels in the sub-anatomy
            masked_img = Image.new('L', (img.size[0], img.size[1]), 225)
            masked_img.paste(0, (x, y, x+w, y+h))
            masked_img = self.transform(masked_img)
            masked_imges.append(masked_img)

        masked_imges = torch.stack(masked_imges)
        sub_anatomies = torch.stack(sub_anatomies)
        sub_anatomy_labels = torch.tensor(sub_anatomy_labels).float()
        img = self.transform(img)
        complete_labels = torch.from_numpy(sample_data[self.df.columns[-15:-1]].to_numpy().astype(np.float32))
        
        return {
            'global_feat': img,
            'node_feat': sub_anatomies,
            'y': sub_anatomy_labels,
            'label_name': sub_anatomy_name,
            'complete_labels': complete_labels,
            'masked_img': masked_imges,
        }

    def __getitem__(self, idx):
        # get anchor sample
        anchor_data = self.df.iloc[idx]
        anchor_data = self.get_sample(anchor_data)
        y = (torch.sum(anchor_data['y'], dim=0) > 0).numpy().astype(int)

        # get positive sample
        postive_indices = np.where((self.nine_labels_list == y).all(axis=1))[0]
        positive_idx = np.random.choice(postive_indices)
        positive_data = self.df.iloc[positive_idx]
        positive_data = self.get_sample(positive_data)

        # get negative sample
        negative_indices = list(set(np.arange(len(self.nine_labels_list))) - set(postive_indices))
        negative_idx = np.random.choice(negative_indices)
        negative_data = self.df.iloc[negative_idx]
        negative_data = self.get_sample(negative_data)

        return {
            'anchor': anchor_data,
            'positive': positive_data,
            'negative': negative_data,
        }