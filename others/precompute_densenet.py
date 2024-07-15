from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--division', type=int, default=0, help='division')
parser.add_argument('--resize', action=argparse.BooleanOptionalAction, help='resize')
args = parser.parse_args()


class CustomDataset(Dataset):
    
    def __init__(self,
                 config: dict,
                 data_prefix: str="/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/scene_tabular/",
                 images_dir: str="/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/mimic-cxr-jpg/2.0.0/files/",
                 split: str="val",
                 to_gen: int=-1,
                 division: int=0,
                 is_resize=True) -> None:
        super(CustomDataset, self).__init__()
        
        self.config = config
        self.is_resize = is_resize
        split = 'validate' if 'val' in split else split
        
        # bounding box files are stored here
        # data_prefix = "/ssd_scratch/cvit/arihanth/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/scene_tabular/"

        # chest x-ray images are stored here
        # images_dir = "/ssd_scratch/cvit/arihanth/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
        
        print(f'Loading {split} data...')
        # self.images = sorted(glob(images_dir + "/**/*.npy", recursive=True))
        # self.is_numpy = len(self.images) > 0
        self.is_numpy = False
        self.images = self.images if self.is_numpy else sorted(glob(images_dir + "/**/*.jpg", recursive=True))

        print(f'Creating {split} dataframe...')
        try:
            self.df = pd.read_pickle(f"{data_prefix}processed_bboxes_{split}.pkl")
        except:
            df = pd.read_pickle(f"{data_prefix}processed_bboxes.pkl")
            df1 = pd.read_csv('data/mimic_cxr_jpg/mimic-cxr-2.0.0-chexpert.csv').fillna(0).replace(-1, 0)
            df2 = pd.read_csv('data/mimic_cxr_jpg/mimic-cxr-2.0.0-split.csv')
            df_split = pd.merge(df1, df2, on=['subject_id', 'study_id'])
            df_split.rename({'dicom_id': 'image_id'}, axis=1, inplace=True)
            df = pd.merge(df_split, df, on=['image_id'])
            self.df = df[df["split"] == split] if split != 'all' else df
            assert len(self.df) > 0, "No data found for the given split"
            del df, df1, df2, df_split
        print(f'Found {len(self.df)} images for {split} split')

        # find all images that have bounding boxes
        self.intersecting_images = sorted(list(set(self.df["image_id"].to_list()) & set([fn.split("/")[-1].split(".")[0] for fn in self.images])))
        print(f'Found {len(self.intersecting_images)} images with bounding box data for {split} split')

        # store only image id and maintain same indexing
        self.only_image_ids = [fn.split("/")[-1].split(".")[0] for fn in self.images]

        # get all bounding box names
        self.bbox_names = sorted(list(set(self.df["bbox_name"].to_list())))

        # decide length of dataset
        self.to_gen = to_gen if to_gen > 0 else len(self.intersecting_images)

        # define my_transform
        self.my_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            # transforms.Normalize((0.5,), (0.5,))
        ])

        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.model.classifier = nn.Identity()

        if division == 0:
            self.to_gen = len(self.intersecting_images)
        elif division == 1:
            self.to_gen = len(self.intersecting_images) // 4
            self.intersecting_images = self.intersecting_images[:self.to_gen]
        elif division == 2:
            self.to_gen = len(self.intersecting_images) // 4
            self.intersecting_images = self.intersecting_images[self.to_gen:2*self.to_gen]
        elif division == 3:
            self.to_gen = len(self.intersecting_images) // 4
            self.intersecting_images = self.intersecting_images[2*self.to_gen:3*self.to_gen]
        elif division == 4:
            self.to_gen = len(self.intersecting_images) // 4
            self.intersecting_images = self.intersecting_images[3*self.to_gen:]
        else:
            raise ValueError('Invalid division')
        print(f'Resizing of images is set to {is_resize}')
        
    def __len__(self):
        return self.to_gen
    
    def __getitem__(self, idx):
        # pick a random image (if req) and find its corresponding image file name
        # rand_idx = torch.randint(0, len(self.intersecting_images), (1,)).item() if self.to_gen != len(self.intersecting_images) else idx
        # cur_img_id = self.intersecting_images[rand_idx]
        cur_img_id = self.intersecting_images[idx]
        req_idx = self.only_image_ids.index(cur_img_id)

        # get patient id and study id
        cur_patient_id = self.images[req_idx].split("/")[-3]
        cur_study_id   = self.images[req_idx].split("/")[-2]
        img_dir        = '/'.join(self.images[req_idx].split("/")[:-1])

        try:
            if self.is_numpy:
                raise NotImplementedError('Numpy not implemented')
                # read image and convert to RGB
                img = np.load(self.images[req_idx])
                img = torch.tensor(img, dtype=torch.float32).repeat(3, 1, 1)
            else:
                # read image and resize
                img = Image.open(self.images[req_idx])
                img = self.my_transform(img)

            # extract graph info about the image
            cur_img_boxes = self.df[self.df["image_id"] == cur_img_id][["x1", "y1", "x2", "y2", "bbox_name"]].to_numpy()
            labels = self.df[self.df["image_id"] == cur_img_id].values.tolist()[0][2:16]
        except:
            raise NotImplementedError(f'Error in reading image p{cur_patient_id}/s{cur_study_id}')
            return self.__getitem__(torch.randint(0, len(self.intersecting_images), (1,)).item())

        # get node features
        node_idx = []
        node_features = []
        track_idx = []
        for i, (x1, y1, x2, y2, bbox_name) in enumerate(cur_img_boxes):
            bbox_id = self.bbox_names.index(bbox_name)
            if bbox_id not in node_idx:
                track_idx.append(i)
                node_idx.append(bbox_id)
                if np.abs(x2-x1) < 1 or np.abs(y2-y1) < 1:
                    node_features.append(torch.zeros(3, 224, 224))
                else:
                    node_features.append(img[:, y1:y2, x1:x2])

        # get adjacency matrix
        edges = []
        track_idx = set(track_idx)
        for i, (x1, y1, x2, y2, _) in enumerate(cur_img_boxes):
            if i not in track_idx:
                continue
            edges.append([])
            for j, (x1_, y1_, x2_, y2_, _) in enumerate(cur_img_boxes):
                if j not in track_idx:
                    continue
                # find intersection over union
                intersection = max(0, min(x2, x2_) - max(x1, x1_)) * max(0, min(y2, y2_) - max(y1, y1_))
                union = np.abs(x2 - x1) * np.abs(y2 - y1) + np.abs(x2_ - x1_) * np.abs(y2_ - y1_) - intersection
                iou = intersection / (union+1e-8)
                edges[-1].append(iou)
        edges = np.array(edges)

        # threshold the adjacency matrix
        threshold = 0.5
        adj_mat = np.where(edges > threshold, 1, 0)

        # get node stuff
        node_idx = torch.tensor(node_idx).long()
        
        # have to pad node features as they are of different sized images
        largest_right  = 224
        largest_bottom = 224
        # largest_right  = max([node_features[i].shape[2] for i in range(len(node_features))])
        # largest_bottom = max([node_features[i].shape[1] for i in range(len(node_features))])

        if self.is_resize:
            node_features = [transforms.Resize((224, 224), antialias=True)(node_features[i]) for i in range(len(node_features))]
        else:
            node_features = [torch.nn.functional.pad(node_features[i], (0, largest_right - node_features[i].shape[2], 0, largest_bottom - node_features[i].shape[1], 0, 0)) for i in range(len(node_features))]
        node_features = torch.stack(node_features)
        
        # find missing node indices, we know a total of 37 nodes are present
        missing_node_idx = list(set(range(len(self.bbox_names))) - set(node_idx.tolist()))
        num_missing = len(missing_node_idx)

        if num_missing:
            node_idx = torch.cat([node_idx, torch.tensor(missing_node_idx).long()])
            node_features = torch.cat([node_features, torch.zeros(num_missing, node_features.shape[1], node_features.shape[2], node_features.shape[3])])
            adj_mat = np.pad(adj_mat, ((0, num_missing), (0, num_missing)))
            edges = np.pad(edges, ((0, num_missing), (0, num_missing)))
        
        # get adjacency matrix and edge features
        adj_mat = torch.tensor(adj_mat).float()
        edges = torch.tensor(edges).float()

        # gamma = 2
        # edges_copy = torch.arange(-1, 1, 2/self.config['edge_dim']).repeat(edges.shape[0], edges.shape[1], 1)
        # edges = torch.exp(-gamma*(torch.pow(edges_copy-edges.unsqueeze(-1), 2)))

        # get mask and labels
        mask = torch.ones(len(node_idx)).bool()
        labels = torch.tensor(labels).float()
        edge_index = adj_mat.nonzero().t().contiguous()

        # with torch.no_grad():
        #     node_features = self.model(node_features).long()
        # labels = labels.unsqueeze(0)

        data_dict = {
            'image': img,
            'node_features': node_features,
            'node_idx': node_idx,
            'adj_mat': adj_mat,
            'edges': edges,
            'labels': labels,
            'img_dir': img_dir,
            'patient_id': cur_patient_id,
            'study_id': cur_study_id,
        }

        return data_dict
    

if __name__ == "__main__":
    division = args.division
    # device = f'cuda'
    device = f'cuda:{division-1 if division > 0 else 0}'
    assert division >= 0 and division <= 4, 'Invalid division'

    train_dataset = CustomDataset(config={'edge_dim': 10}, split='train', division=division, is_resize=args.resize)
    val_dataset   = CustomDataset(config={'edge_dim': 10}, split='val', division=division, is_resize=args.resize)
    test_dataset  = CustomDataset(config={'edge_dim': 10}, split='test', division=division, is_resize=args.resize)

    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Identity()
    model = model.to(device)

    val_loader   = DataLoader(val_dataset, batch_size=1, num_workers=8)
    test_loader  = DataLoader(test_dataset, batch_size=1, num_workers=8)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=8)

    for loader, split in zip([val_loader, test_loader, train_loader], ['validate', 'test', 'train']):
        for i, data in enumerate(tqdm(loader, desc=split)):
            
            image = data['image']
            # node_features = data['node_features'].squeeze(0)
            # node_idx = data['node_idx'].squeeze(0)
            # adj_mat = data['adj_mat'].squeeze(0)
            # edges = data['edges'].squeeze(0)
            # labels = data['labels'].squeeze(0)
            img_dir = data['img_dir'][0]
            p_id = data['patient_id'][0]
            s_id = data['study_id'][0]

            with torch.no_grad():
                # node_features = model(node_features.to(device)).long().to('cpu')
                global_feature = model(image.to(device)).long().to('cpu').squeeze(0)

            np.save(f'/scratch/arihanth.srikar/{split}/{p_id}_{s_id}_image.npy', image.squeeze(0).numpy())
            np.save(f'/scratch/arihanth.srikar/{split}/{p_id}_{s_id}_global_feature.npy', global_feature.numpy())

            # if args.resize:
            #     np.save(f'/scratch/arihanth.srikar/{split}/{p_id}_{s_id}_node_features_resize.npy', node_features.numpy())
            # else:
            #     np.save(f'/scratch/arihanth.srikar/{split}/{p_id}_{s_id}_node_features.npy', node_features.numpy())
            # np.save(f'/scratch/arihanth.srikar/{split}/{p_id}_{s_id}_node_idx.npy', node_idx.numpy())
            # np.save(f'/scratch/arihanth.srikar/{split}/{p_id}_{s_id}_adj_mat.npy', adj_mat.numpy())
            # np.save(f'/scratch/arihanth.srikar/{split}/{p_id}_{s_id}_edges.npy', edges.numpy())
            # np.save(f'/scratch/arihanth.srikar/{split}/{p_id}_{s_id}_labels.npy', labels.numpy())
