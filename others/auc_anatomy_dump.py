import warnings
warnings.filterwarnings("ignore")

import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import auc as compute_auc
from glob import glob
import torch
import torch.nn as nn
from tqdm import tqdm
import json

from dataloader.anaxnet import OcclusionDataset
from model.graph_transformer import CustomModel as GTModel

# get arguments from the command line without argparse
args = sys.argv
gpu_id = int(args[1])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = f'cuda:{gpu_id}'

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

def cosine_similarity(a, b):
    a = a.reshape(-1) if len(a.shape) > 1 else a
    b = b.reshape(-1) if len(b.shape) > 1 else b
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_all_retrieval_info(
        df: pd.DataFrame, 
        query_id: int, 
        retrieved_ids: list,
        model: nn.Module, 
        anatomy_names: list=anatomy_names):
    
    test_dataset = OcclusionDataset(df, split='test')
    
    # get query info for a model
    original_sample = test_dataset[query_id]
    inp = original_sample['node_feat'].unsqueeze(0).to(device)
    y = original_sample['y']
    y_9 = torch.sum(y, dim=0) > 0
    relevant_anatomies = [anatomy_names[i] for i, y_row in enumerate(y) if y_row.sum() > 0]
    relevant_diseases = [nine_class_labels[i] for i, y_row in enumerate(y_9) if y_row]
    disease_dict = {disease: [anat] for (anat, lbl) in zip(anatomy_names, original_sample['y']) for (disease, l) in zip(nine_class_labels, lbl) if l}
    disease_dict = {k: list(set(v)) for k, v in disease_dict.items() if len(list(set(v))) > 0}

    lookup_dict = {'q': {'img': original_sample['global_feat'], 'mask': original_sample['masked_img'], 
                                'sub_anatomies': original_sample['node_feat'], 'sub_anatomy_masks': original_sample['sub_anatomy_masks']}}
    lookup_dict['q']['emb'] = model.retrieval_pass(inp).detach().cpu().squeeze()
    lookup_dict['q']['anatomies'] = relevant_anatomies
    lookup_dict['q']['diseases'] = relevant_diseases
    lookup_dict['q']['anatomy_for_disease'] = disease_dict
    
    for r_id, retrieved_id in enumerate(retrieved_ids):
        # get original info for a model
        original_sample = test_dataset[retrieved_id]
        inp = original_sample['node_feat'].unsqueeze(0).to(device)
        y = original_sample['y']
        y_9 = torch.sum(y, dim=0) > 0
        relevant_anatomies = [anatomy_names[i] for i, y_row in enumerate(y) if y_row.sum() > 0]
        relevant_diseases = [nine_class_labels[i] for i, y_row in enumerate(y_9) if y_row]
        disease_dict = {disease: [anat] for (anat, lbl) in zip(anatomy_names, original_sample['y']) for (disease, l) in zip(nine_class_labels, lbl) if l}
        disease_dict = {k: list(set(v)) for k, v in disease_dict.items() if len(list(set(v))) > 0}

        lookup_dict[f'r{r_id+1}'] = {'img': original_sample['global_feat'], 'mask': original_sample['masked_img'], 
                                    'sub_anatomies': original_sample['node_feat'], 'sub_anatomy_masks': original_sample['sub_anatomy_masks']}
        lookup_dict[f'r{r_id+1}']['emb'] = model.retrieval_pass(inp).detach().cpu().squeeze()
        lookup_dict[f'r{r_id+1}']['anatomies'] = relevant_anatomies
        lookup_dict[f'r{r_id+1}']['diseases'] = relevant_diseases
        lookup_dict[f'r{r_id+1}']['anatomy_for_disease'] = disease_dict

        # get occluded info for a model
        for anatomy in anatomy_names:
            # initialise dataset with occluded anatomy
            occluded_dataset = OcclusionDataset(df, split='test', occlude_anatomy=[anatomy])
            occluded_sample = occluded_dataset[retrieved_id]
            inp = occluded_sample['node_feat'].unsqueeze(0).to(device)
            
            # save the occluded anatomy's features and mask
            lookup_dict[f'r{r_id+1}'][anatomy] = {'img': occluded_sample['global_feat'], 'mask': occluded_sample['masked_img'], 
                                    'sub_anatomies': original_sample['node_feat'], 'sub_anatomy_masks': occluded_sample['sub_anatomy_masks']}
            lookup_dict[f'r{r_id+1}'][anatomy]['emb'] = model.retrieval_pass(inp).detach().cpu().squeeze()

    return lookup_dict

def auc_anatomy_occlusion(
        lookup_dict: dict,
        model: nn.Module, 
        anatomy_names: list=anatomy_names,
        resolution: int=11,
        res_arr: np.array=None):

    query_emb = lookup_dict['q']['emb']

    for k in lookup_dict.keys():
        if k == 'q':
            continue
        
        for anatomy in anatomy_names:
            # create resolution number of masks
            sim_diffs = []
            node_feat_batched = []
            if res_arr is None:
                res_arr = np.linspace(0, 1, resolution)
            for res in res_arr:
                # get the sub anatomy features and mask
                sub_anatomy_imgs = lookup_dict[k]['sub_anatomies']
                sub_anatomy_masks = (lookup_dict[k][anatomy]['sub_anatomy_masks'] > 0).float()
                sub_anatomy_inv_masks = (lookup_dict[k][anatomy]['sub_anatomy_masks'] == 0).float() * res
                node_feat = sub_anatomy_imgs * sub_anatomy_masks + sub_anatomy_imgs * sub_anatomy_inv_masks
                node_feat_batched.append(node_feat)
                
            # get the embeddings
            inp = torch.stack(node_feat_batched).to(device)
            embeddings = model.retrieval_pass(inp).detach().cpu().squeeze()
                
            # get cosine similarity
            orig_sim = max(0, cosine_similarity(query_emb, lookup_dict[k]['emb']))
            for emb in embeddings:
                sim = max(0, cosine_similarity(query_emb, emb))
                diff = np.abs((orig_sim - sim) / orig_sim)
                sim_diffs.append(diff)
            auc = compute_auc(res_arr, sim_diffs)
            lookup_dict[k][anatomy]['auc'] = auc
            lookup_dict[k][anatomy]['emb'] = emb

    return lookup_dict

config = {
    'lr': 0.0001,
    'num_layers': 2,
    'graph_importance': 1.0,
    'dropout': 0.0,
    'fully_connected': True,
    'contrastive': False,
    'concat_global_feature': False,
    'image_featuriser': 'resnet',
    'multiscale': False,
    'is_global_feat': False,
    'cls': False,
    'matryoshka': False,
    'rel_pos': False,
    'abs_pos': True,
    'accept_edges': True,
    'residual_type': 2,
    'task': 'graph_transformer',
    'run': 'best_config_abs_pos_with_edges',
}
model_paths = sorted(glob(f'/home/ssd_scratch/users/arihanth.srikar/checkpoints/mimic-cxr/{config["run"]}/*.ckpt'))
model_path = [md_name for md_name in model_paths if 'auc' in md_name][-1]
best_model = GTModel.load_from_checkpoint(model_path, config=config)
best_model = best_model.eval()
best_model = best_model.to(device)

for p in best_model.parameters():
    p.requires_grad = False

print("Loading dataset")
df = pd.read_json('/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/mimic_coco_filtered.json')
temp_df = pd.read_csv('data/mimic_cxr_jpg/mimic-cxr-2.0.0-final.csv')
temp_df.rename(columns={'dicom_id': 'image_id'}, inplace=True)
df = df.merge(temp_df, on='image_id', how='left')
print("Dataset loaded")

topk = 10
df_gt = pd.read_csv(f'results/graph_transformer/best_config_abs_pos_with_edges/occluded/orig/9_classes_auc_no_global_feat_top_{topk}_cosine_.txt')

df_consider = df_gt[['query_indices', 'target_indices']].drop_duplicates().copy()
df_consider = df_consider.reset_index(drop=True).sort_values(by='query_indices')

# cached_files = sorted(glob('/scratch/arihanth.srikar/dump_imp_output/auc_analysis/lookup_dicts/*.pkl'))
# cached_ids = set([int(f.split('/')[-1].split('.')[0]) for f in cached_files])

dir_path = '/scratch/arihanth.srikar/dump_imp_output/auc_analysis/lookup_dicts'
os.makedirs(dir_path, exist_ok=True)

t = len(df_consider)
if gpu_id == 0:
    l = 0
    r = t//3
elif gpu_id == 1:
    l = t//3
    r = l + t//3
elif gpu_id == 2:
    l = int(2*t//3)
    r = t
else:
    l = 0
    r = t
# if gpu_id == 0:
#     l = 0
#     r = t//4
# elif gpu_id == 1:
#     l = t//4
#     r = l + t//4
# elif gpu_id == 2:
#     l = t//2
#     r = l + t//4
# elif gpu_id == 3:
#     l = int(3*t//4)
#     r = t
# else:
#     l = 0
#     r = t
print(f"GPU {gpu_id} processing from {l} to {r}")


all_aucs = []
for _, entry in tqdm(df_consider.iloc[l:r].iterrows(), total=r-l):
    q = entry['query_indices']
    rs = eval(entry['target_indices'])

    lookup_dict = get_all_retrieval_info(df, q, rs, best_model, anatomy_names)
    lookup_dict = auc_anatomy_occlusion(lookup_dict, best_model, anatomy_names, resolution=11)

    auc_list = []
    for k, v in lookup_dict.items():
        if k == 'q':
            continue
        for anatomy in anatomy_names:
            auc = lookup_dict[k][anatomy]['auc']
            auc_list.append(auc)
    auc_list = np.array(auc_list)
    all_aucs.append(auc_list[np.newaxis, :])
    np.save(f'{dir_path}/{q}.npy', auc_list)

all_aucs = np.concatenate(all_aucs, axis=0)
np.save(f'{dir_path}/gpu_{gpu_id}.npy', all_aucs)
print(f"GPU {gpu_id} done")
