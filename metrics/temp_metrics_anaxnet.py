import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import numpy as np
import pandas as pd
from glob import glob
import torch
from torch_geometric.loader import DataLoader
import argparse

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from dataloader.anaxnet import OcclusionDataset
from common_metrics import compute_metrics

parser = argparse.ArgumentParser()

parser.add_argument('--emb_dim', type=int, default=512, help='embedding dimension')
parser.add_argument('--edge_dim', type=int, default=32, help='edge embedding dimension')
parser.add_argument('--num_classes', type=int, default=14, help='number of classes')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--grad_accum', type=int, default=1, help='gradient accumulation steps')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
parser.add_argument('--edge_index', type=int, default=128, help='edge index')
parser.add_argument('--graph_importance', type=float, default=0.2, help='graph importance bewteen 0 and 1')
parser.add_argument('--is_global_feat', action=argparse.BooleanOptionalAction, help='add global feature')
parser.add_argument('--concat_global_feature', action=argparse.BooleanOptionalAction, help='Concat global feature for retrieval')
parser.add_argument('--matryoshka', action=argparse.BooleanOptionalAction, help='Concat global feature for retrieval')
parser.add_argument('--contrastive', action=argparse.BooleanOptionalAction, help='Train the embedding model in a contrastive manner')
parser.add_argument('--hash_bits', type=int, default=64, help='number of bits for hashing')
parser.add_argument('--pool', type=str, default='attn', help='pooling method')
parser.add_argument('--minimalistic', action=argparse.BooleanOptionalAction, help='Train using only the graph')
parser.add_argument('--prune', action=argparse.BooleanOptionalAction, help='Train using only the graph')
parser.add_argument('--fully_connected', action=argparse.BooleanOptionalAction, help='Fully connected graph')
parser.add_argument('--abs_pos', action=argparse.BooleanOptionalAction, help='Absolute learnt positional embeddings')
parser.add_argument('--rel_pos', action=argparse.BooleanOptionalAction, help='Relative positional embeddings')
parser.add_argument('--lr_scheduler', type=str, default='plateau', help='learning rate scheduler')
parser.add_argument('--image_featuriser', type=str, default='resnet', help='pick image featuriser from resnet, densenet')
parser.add_argument('--cls', action=argparse.BooleanOptionalAction, help='Use CLS token')
parser.add_argument('--multiscale', action=argparse.BooleanOptionalAction, help='multiscale resnet')
parser.add_argument('--accept_edges', action=argparse.BooleanOptionalAction, help='accept edges')
parser.add_argument('--residual_type', type=int, default=2, help='1: local gated residue, 2: global gated residue, 3: dense gated residue')
parser.add_argument('--num_nodes', type=int, default=18, help='number of nodes')
parser.add_argument('--naren', action=argparse.BooleanOptionalAction, help='modified metrics for naren')
parser.add_argument('--view', type=str, default='None', help='Consider AP for metrics else PA')
parser.add_argument('--gender', type=str, default='None', help='Consider male or female or both')
parser.add_argument('--lower_age_limit', type=int, default=0, help='lower age limit')
parser.add_argument('--upper_age_limit', type=int, default=100, help='lower age limit')

parser.add_argument('--task', type=str, default='mimic-cxr-emb', help='model name')
parser.add_argument('--save_dir', type=str, default='/home/ssd_scratch/users/arihanth.srikar/checkpoints', help='save directory')
parser.add_argument('--entity', type=str, default='arihanth', help='wandb entity name')
parser.add_argument('--project', type=str, default='mimic-cxr', help='wandb project name')
parser.add_argument('--run', type=str, default='test', help='wandb run name')
parser.add_argument('--file_name', type=str, default='graph_metrics.txt', help='save to file')
parser.add_argument('--ablation', action=argparse.BooleanOptionalAction, help='Perform ablation study')

parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0], help='list of gpu ids')
parser.add_argument('--set_precision', action=argparse.BooleanOptionalAction, help='set precision')
parser.add_argument('--log', action=argparse.BooleanOptionalAction, help='log to wandb')
parser.add_argument('--compile', action=argparse.BooleanOptionalAction, help='compile model')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
parser.add_argument('--train', action=argparse.BooleanOptionalAction, help='train model')

parser.add_argument('--validate_every', type=int, default=1000, help='train for n epochs')
parser.add_argument('--validate_for', type=int, default=200, help='validate for n epochs')

config = vars(parser.parse_args())

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
    'pneumonia',
    ]

eighteen_class_labels = [
    'right lung opacity', 
    'right pleural effusion', 
    'right atelectasis', 
    'right enlarged cardiac silhouette',
    'right pulmonary edema/hazy opacity', 
    'right pneumothorax', 
    'right consolidation', 
    'right fluid overload/heart failure', 
    'right pneumonia',
    'left lung opacity', 
    'left pleural effusion', 
    'left atelectasis', 
    'left enlarged cardiac silhouette',
    'left pulmonary edema/hazy opacity', 
    'left pneumothorax', 
    'left consolidation', 
    'left fluid overload/heart failure', 
    'left pneumonia',
    ]

consider_class_labels = [
    'right lung opacity', 
    'right pleural effusion', 
    'right atelectasis', 
    'right pulmonary edema/hazy opacity', 
    'right pneumothorax', 
    'right consolidation', 
    'right fluid overload/heart failure', 
    'right pneumonia',
    'left lung opacity', 
    'left pleural effusion', 
    'left atelectasis', 
    'left pulmonary edema/hazy opacity', 
    'left pneumothorax', 
    'left consolidation', 
    'left fluid overload/heart failure', 
    'left pneumonia',
    'enlarged cardiac silhouette',
    ]

thirtysix_class_labels = [
    'right lung opacity', 
    'right pleural effusion', 
    'right atelectasis', 
    'right enlarged cardiac silhouette',
    'right pulmonary edema/hazy opacity', 
    'right pneumothorax', 
    'right consolidation', 
    'right fluid overload/heart failure', 
    'right pneumonia',
    'left lung opacity', 
    'left pleural effusion', 
    'left atelectasis', 
    'left enlarged cardiac silhouette',
    'left pulmonary edema/hazy opacity', 
    'left pneumothorax', 
    'left consolidation', 
    'left fluid overload/heart failure', 
    'left pneumonia',
    'middle lung opacity', 
    'middle pleural effusion', 
    'middle atelectasis', 
    'middle enlarged cardiac silhouette',
    'middle pulmonary edema/hazy opacity', 
    'middle pneumothorax', 
    'middle consolidation', 
    'middle fluid overload/heart failure', 
    'middle pneumonia',
    'lung opacity', 
    'pleural effusion', 
    'atelectasis', 
    'enlarged cardiac silhouette',
    'pulmonary edema/hazy opacity', 
    'pneumothorax', 
    'consolidation', 
    'fluid overload/heart failure', 
    'pneumonia',
    ]

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
    'trachea',]


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    if config['task'] == 'anaxnet':
        from model.anaxnet import CustomModel
    elif 'anaxnet_attn_multilayer' in config['task']:
        from model.anaxnet_attn_multilayer import CustomModel
    elif 'anaxnet_attn' in config['task']:
        from model.anaxnet_attn import CustomModel
    elif 'anaxnet_custom' in config['task']:
        from model.anaxnet_custom import CustomModel
    elif 'graph_benchmark' in config['task']:
        from model.graph_benchmark import CustomModel
    elif 'xfactor' in config['task']:
        from model.xfactor import CustomModel
    elif 'graph_transformer' in config['task']:
        from model.graph_transformer import CustomModel
    elif 'vanilla_transformer' in config['task']:
        from model.vanilla_transformer import CustomModel
    else:
        raise NotImplementedError

    device = 'cuda'
    seed_everything(42)

    print("Loading dataset")
    try:
        df = pd.read_json('/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/mimic_coco_filtered.json')
    except:
        df = pd.read_json('/scratch/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/mimic_coco_filtered.json')
    temp_df = pd.read_csv('data/mimic_cxr_jpg/mimic-cxr-2.0.0-final.csv')
    temp_df.rename(columns={'dicom_id': 'image_id'}, inplace=True)
    df = df.merge(temp_df, on='image_id', how='left')
    
    view = None
    sex = None
    age = None
    if config['naren']:
        print('Naren metrics')
        if config['view'] == 'AP' or config['view'] == 'PA':
            df_meta = pd.read_csv('data/mimic_cxr_jpg/mimic-cxr-2.0.0-metadata.csv')
            view = 'AP' if config['view'] == 'AP' else 'PA'
            df_meta = df_meta[df_meta['ViewPosition'].str.contains(view, case=False, na=False)]
            df = df[df['image_id'].isin(df_meta['dicom_id'])]
        
        if config['gender'] == 'M' or config['gender'] == 'F':
            df_meta = pd.read_csv('data/mimic_cxr_jpg/patients.csv')
            sex = 'M' if config['gender'] == 'M' else 'F'
            df_meta = df_meta[df_meta['gender'].str.contains(sex, case=False, na=False)]
            df = df[df['subject_id'].isin(df_meta['subject_id'])]

        df_meta = pd.read_csv('data/mimic_cxr_jpg/patients.csv')
        df_meta = df_meta[(df_meta['anchor_age'] >= config['lower_age_limit']) & (df_meta['anchor_age'] < config['upper_age_limit'])]
        df = df[df['subject_id'].isin(df_meta['subject_id'])]
        age = f'{config["lower_age_limit"]}-{config["upper_age_limit"]}'
    print("Dataset loaded")

    # load model from checkpoint
    model_paths = sorted(glob(f'/home/ssd_scratch/users/arihanth.srikar/checkpoints/mimic-cxr/{config["run"]}/*.ckpt'))
    model_paths = sorted(glob(f'/scratch/arihanth.srikar/checkpoints/mimic-cxr/{config["run"]}/*.ckpt')) if len(model_paths) == 0 else model_paths
    assert len(model_paths) > 0, 'No model found'

    combined_anatomy_names = [
        ['right lung', 'left lung'],
        ['right apical zone', 'left apical zone'],
        ['right upper lung zone', 'left upper lung zone'],
        ['right mid lung zone', 'left mid lung zone'],
        ['right lower lung zone', 'left lower lung zone'],
        ['right hilar structures', 'left hilar structures'],
        ['right costophrenic angle', 'left costophrenic angle'],
        ['mediastinum', 'upper mediastinum'],
        ['cardiac silhouette'],
        ['trachea']]
    consider_anatomies = [[]] if not config['ablation'] else [[]] + [a_list for a_list in combined_anatomy_names]
    
    # for model_criteria in ['randomly_initialized', 'auc', 'mAP'] if not config['ablation'] else ['auc']:
    for model_criteria in ['auc'] if not config['ablation'] else ['auc']:
        # load model from checkpoint
        try:
            print(f'Loading best model based on {model_criteria}')
            model_path = [md_name for md_name in model_paths if model_criteria in md_name][-1]
            model = CustomModel.load_from_checkpoint(model_path, config=config)
            print(f'Loading best model based on {model_criteria}')
        except:
            if model_criteria != 'randomly_initialized': continue
            model = CustomModel(config)
            print(f'Loading {model_criteria} model')

        if config['prune']:
            model.cnn.layer4 = torch.nn.Identity()
            model.cnn.fc = torch.nn.Identity()
        model = model.to(device)
        model = model.eval()

        original_emb = None
        original_labels = None

        for anatomy_list in consider_anatomies:
            config['occluded_anatomies'] = '-'.join(anatomy_list).replace(' ', '_')
            test_dataset = OcclusionDataset(df, split='test', occlude_anatomy=anatomy_list)
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
            print(f'Occulding {anatomy_list} from the test set using {model_criteria} model')

            all_emb = []
            all_labels = []
            positioned_labels = []
            
            with torch.no_grad():
                for i, batch in enumerate(tqdm(test_loader)):
                    # move to device
                    node_feats = batch['node_feat'].to(device)
                    global_feat = batch['global_feat'].to(device)
                    y = torch.sum(batch['y'], dim=1) > 0
                    
                    # compute embedding
                    emb = model.retrieval_pass(node_feats, global_feat).detach().cpu()
                    # emb = model.retrieval_pass(node_feats, global_feat, return_anatomy=True).detach().cpu()
                    
                    # store embeddings and labels for building the datastore
                    all_emb.append(emb.numpy())
                    all_labels.append(y.numpy())
                    
                    # modified right, left, middle, and all structure labels
                    only_right = torch.sum(batch['y'][:, :7], dim=1) > 0
                    only_left = torch.sum(batch['y'][:, 7:14], dim=1) > 0
                    only_middle = torch.sum(batch['y'][:, 14:], dim=1) > 0
                    all_structures = torch.sum(batch['y'], dim=1) > 0
                    
                    consider_indices = [thirtysix_class_labels.index(lbl_name) for lbl_name in consider_class_labels]
                    consider_labels = torch.cat([only_right, only_left, only_middle, all_structures], dim=1)[:, consider_indices]
                    positioned_labels.append(consider_labels.numpy())

            # cache for later use
            all_emb = np.concatenate(all_emb)
            all_labels = np.concatenate(all_labels)
            positioned_labels = np.concatenate(positioned_labels)
            print(all_emb.shape, all_labels.shape)
            np.save(f'/tmp/{config["run"]}_emb_{model_criteria}{"_"+"-".join(anatomy_list) if len(anatomy_list) else ""}.npy', all_emb)
            np.save(f'/tmp/{config["run"]}_labels_{model_criteria}{"_"+"-".join(anatomy_list) if len(anatomy_list) else ""}.npy', all_labels)
            np.save(f'/tmp/{config["run"]}_positioned_labels_{model_criteria}{"_"+"-".join(anatomy_list) if len(anatomy_list) else ""}.npy', positioned_labels)

            # exit()

            if len(anatomy_list) == 0:
                original_emb = all_emb.copy()
                original_labels = all_labels.copy()
            
            # print('Loading embeddings and labels...')
            # all_emb = np.load(f'/tmp/{config["run"]}_emb_{model_criteria}.npy')
            # all_labels = np.load(f'/tmp/{config["run"]}_labels_{model_criteria}.npy')
            # positioned_labels = np.load(f'/tmp/{config["run"]}_positioned_labels_{model_criteria}.npy')
            # original_emb = all_emb.copy()
            # original_labels = all_labels.copy()

            for top_k in [3, 5, 10] if not config['ablation'] else [10]:
                # compute metrics
                compute_metrics(config, model_criteria, all_emb, all_labels, nine_class_labels, top_k=top_k, dist_metric='cosine', query_emb=original_emb, skip=True, naren=config['naren'], view=view, sex=sex, age=age)
                compute_metrics(config, model_criteria, all_emb, all_labels, nine_class_labels, top_k=top_k, dist_metric='cosine', query_emb=original_emb, skip=False, naren=config['naren'], view=view, sex=sex, age=age)
                
                # right, left, and middle structure disease metrics
                compute_metrics(config, model_criteria, all_emb, positioned_labels, consider_class_labels, top_k=top_k, dist_metric='cosine', query_emb=original_emb, skip=True, naren=config['naren'], view=view, sex=sex, age=age)
                compute_metrics(config, model_criteria, all_emb, positioned_labels, consider_class_labels, top_k=top_k, dist_metric='cosine', query_emb=original_emb, skip=False, naren=config['naren'], view=view, sex=sex, age=age)
