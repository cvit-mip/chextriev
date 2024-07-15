import numpy as np
from glob import glob

import torch
from torch.utils.data import Dataset


class GTPreprocessed(Dataset):
    def __init__(self, config: dict, data_dir: str='/scratch/arihanth.srikar', split: str='val', threshold: float=0.5):
        super().__init__()
        
        split = 'validate' if split == 'val' else split
        self.split = split
        self.threshold = threshold
        
        self.req_boxes = sorted([
            "right lung", "right apical zone", "right upper lung zone", "right mid lung zone", 
            "right lower lung zone", "right hilar structures", "right costophrenic angle", "left lung", "left apical zone",
            "left upper lung zone", "left mid lung zone", "left lower lung zone", "left hilar structures", 
            "left costophrenic angle", "mediastinum", "upper mediastinum", "cardiac silhouette", "trachea",
            ])
        self.disease_list = [
            'Lung Opacity', 'Pleural Effusion', 'Atelectasis', 'Enlarged Cardiomediastinum',
            'Edema', 'Pneumothorax', 'Consolidation', 'Pneumonia'
            ]
        bbox_names = ['abdomen', 'aortic arch', 'cardiac silhouette', 'carina', 'cavoatrial junction', 
                      'descending aorta', 'left apical zone', 'left cardiac silhouette', 'left cardiophrenic angle', 
                      'left clavicle', 'left costophrenic angle', 'left hemidiaphragm', 'left hilar structures', 
                      'left lower lung zone', 'left lung', 'left mid lung zone', 'left upper abdomen', 
                      'left upper lung zone', 'mediastinum', 'right apical zone', 'right atrium', 
                      'right cardiac silhouette', 'right cardiophrenic angle', 'right clavicle', 
                      'right costophrenic angle', 'right hemidiaphragm', 'right hilar structures', 'right lower lung zone', 
                      'right lung', 'right mid lung zone', 'right upper abdomen', 'right upper lung zone', 
                      'spine', 'svc', 'trachea', 'upper mediastinum']
        self.req_idxs = set([bbox_names.index(box) for box in self.req_boxes])
        
        self.data_dir = data_dir
        self.image_names = ['_'.join(s.split('/')[-1].split('_')[:2]) for s in sorted(glob(f'/scratch/arihanth.srikar/{split}/*edges.npy'))]

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        prefix = self.image_names[idx]
        node_feats = np.load(f'{self.data_dir}/{self.split}/{prefix}_node_features_resize.npy')
        edges = np.load(f'{self.data_dir}/{self.split}/{prefix}_edges.npy')
        # adj_mat = np.load(f'{self.data_dir}/{self.split}/{prefix}_adj_mat.npy')
        # node_idx = np.load(f'{self.data_dir}/{self.split}/{prefix}_node_idx.npy')
        labels = np.load(f'{self.data_dir}/{self.split}/{prefix}_labels.npy')
        
        adj_mat = np.where(edges > self.threshold, 1, 0)[list(self.req_idxs)][:, list(self.req_idxs)]
        
        node_feats = torch.tensor(node_feats, dtype=torch.float)[list(self.req_idxs)]
        adj_mat = torch.tensor(adj_mat, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.float)

        if node_feats.shape[0] != len(self.req_idxs):
            print(f'Error in {prefix} with shape {node_feats.shape[0]}')
            return self.get(torch.randint(0, self.len(), (1,)).item())
        
        return node_feats, adj_mat, labels
