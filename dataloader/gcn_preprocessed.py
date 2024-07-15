from glob import glob
import numpy as np
import torch
from torch_geometric.data import Dataset, Data


class GCNPreprocessed(Dataset):
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

    def get_edge_list(self, adj_mat: np.ndarray, req_idxs: set, threshold: int=0.5):
        edge_list = []
        edge_weights = []
        index_corr = {val: i for i, val in enumerate(list(req_idxs))}
        for i in range(adj_mat.shape[0]):
            if i not in req_idxs:
                continue
            for j in range(adj_mat.shape[1]):
                if j not in req_idxs:
                    continue
                if adj_mat[i][j] >= threshold:
                    edge_list.append([index_corr[i], index_corr[j]])
                    edge_weights.append(adj_mat[i][j])
        return np.array(edge_list), np.array(edge_weights)
    
    def len(self):
        return len(self.image_names)
    
    def get(self, idx):
        prefix = self.image_names[idx]
        node_feats = np.load(f'{self.data_dir}/{self.split}/{prefix}_node_features_resize.npy')
        edges = np.load(f'{self.data_dir}/{self.split}/{prefix}_edges.npy')
        # adj_mat = np.load(f'{self.data_dir}/{self.split}/{prefix}_adj_mat.npy')
        # node_idx = np.load(f'{self.data_dir}/{self.split}/{prefix}_node_idx.npy')
        labels = np.load(f'{self.data_dir}/{self.split}/{prefix}_labels.npy')
        
        # adj_mat = np.where(edges > self.threshold, 1, 0)
        # adj_mat = np.concatenate([arr[..., np.newaxis] for arr in adj_mat.nonzero()], axis=1)
        
        edge_list, edge_weights = self.get_edge_list(edges, self.req_idxs, threshold=self.threshold)

        node_feats = torch.tensor(node_feats, dtype=torch.float)[list(self.req_idxs)]
        edge_list = torch.tensor(edge_list.T, dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.float).unsqueeze(0)

        if node_feats.shape[0] != len(self.req_idxs):
            print(f'Error in {prefix} with shape {node_feats.shape[0]}')
            return self.get(torch.randint(0, self.len(), (1,)).item())

        sample = Data(x=node_feats, edge_index=edge_list, edge_attr=edge_attr, y=labels)
        
        return sample
