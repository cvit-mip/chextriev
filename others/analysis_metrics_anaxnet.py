import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import numpy as np
import pandas as pd
from glob import glob
import torch
from torch_geometric.loader import DataLoader
import argparse

from dataloader.anaxnet import CustomDataset
from common_metrics import compute_occluded_metrics

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
parser.add_argument('--graph_importance', type=float, default=0.2, help='graph importance bewteen 0 and 1')
parser.add_argument('--is_global_feat', action=argparse.BooleanOptionalAction, help='add global feature')
parser.add_argument('--concat_global_feature', action=argparse.BooleanOptionalAction, help='Concat global feature for retrieval')

parser.add_argument('--task', type=str, default='mimic-cxr-emb', help='model name')
parser.add_argument('--save_dir', type=str, default='/home/ssd_scratch/users/arihanth.srikar/checkpoints', help='save directory')
parser.add_argument('--entity', type=str, default='arihanth', help='wandb entity name')
parser.add_argument('--project', type=str, default='mimic-cxr', help='wandb project name')
parser.add_argument('--run', type=str, default='test', help='wandb run name')
parser.add_argument('--file_name', type=str, default='graph_metrics.txt', help='save to file')

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
    'pneumonia']

anatomy_names = [
    "right lung", "right apical zone", "right upper lung zone", "right mid lung zone", 
    "right lower lung zone", "right hilar structures", "right costophrenic angle", "left lung", "left apical zone",
    "left upper lung zone", "left mid lung zone", "left lower lung zone", "left hilar structures", 
    "left costophrenic angle", "mediastinum", "upper mediastinum", "cardiac silhouette"]


if __name__ == '__main__':

    if config['task'] == 'anaxnet':
        from model.anaxnet import CustomModel
    elif 'anaxnet_attn_multilayer' in config['task']:
        from model.anaxnet_attn_multilayer import CustomModel
    elif 'anaxnet_attn' in config['task']:
        from model.anaxnet_attn import CustomModel
    elif 'anaxnet_custom' in config['task']:
        from model.anaxnet_custom import CustomModel
    else:
        raise NotImplementedError

    device = 'cuda'

    print("Loading dataset")
    try:
        df = pd.read_json('/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/mimic_coco_filtered.json')
    except:
        df = pd.read_json('/scratch/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/mimic_coco_filtered.json')
    temp_df = pd.read_csv('data/mimic_cxr_jpg/mimic-cxr-2.0.0-final.csv')
    temp_df.rename(columns={'dicom_id': 'image_id'}, inplace=True)
    df = df.merge(temp_df, on='image_id', how='left')
    print("Dataset loaded")

    test_dataset = CustomDataset(df, split='test')
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # load model from checkpoint
    model_paths = sorted(glob(f'/home/ssd_scratch/users/arihanth.srikar/checkpoints/mimic-cxr/{config["run"]}/*.ckpt'))
    if len(model_paths) == 0: model_paths = sorted(glob(f'/scratch/arihanth.srikar/checkpoints/mimic-cxr/{config["run"]}/*.ckpt'))
    
    for model_criteria in ['randomly_initialized', 'auc']:

        # load model from checkpoint
        try:
            model_path = [md_name for md_name in model_paths if model_criteria in md_name][-1]
            print(f'Loading best model based on {model_criteria}')
            model = CustomModel.load_from_checkpoint(model_path, config=config)
        except:
            if model_criteria != 'randomly_initialized': continue
            print(f'Loading {model_criteria} model')
            model = CustomModel(config)

        model = model.to(device)
        model = model.eval()

        gt_emb = []
        gt_labels = []
        anatomy_embs = [[] for _ in range(len(anatomy_names))]
        
        with torch.no_grad():
            for _, batch in enumerate(tqdm(test_loader)):
                node_feats = batch['node_feat'].to(device)
                global_feat = batch['global_feat'].to(device)
                y = torch.sum(batch['y'], dim=1) > 0
                emb = model.retrieval_pass(node_feats, global_feat, config['concat_global_feature'])
                gt_emb.append(emb.detach().cpu().numpy())
                gt_labels.append(y.numpy())
                for i, anatomy_name in enumerate(anatomy_names):
                    occluded_node_feats = node_feats.clone()
                    occluded_node_feats[:, i] = 0
                    occluded_emb = model.retrieval_pass(occluded_node_feats, global_feat, config['concat_global_feature'])
                    anatomy_embs[i].append(occluded_emb.detach().cpu().numpy())

        # cache for later use
        gt_emb = np.concatenate(gt_emb)
        gt_labels = np.concatenate(gt_labels)
        anatomy_embs = [np.concatenate(anatomy_emb) for anatomy_emb in anatomy_embs]
        print(gt_emb.shape, gt_labels.shape)
        np.save(f'/tmp/analysis_{config["task"]}_emb_{config["run"]}_{model_criteria}.npy', gt_emb)
        np.save(f'/tmp/analysis_{config["task"]}_labels_{config["run"]}_{model_criteria}.npy', gt_labels)
        for i, anatomy_name in enumerate(anatomy_names):
            np.save(f'/tmp/analysis_{config["task"]}_emb_{config["run"]}_{model_criteria}_{anatomy_name}.npy', anatomy_embs[i])

        # print('Loading embeddings and labels...')
        # gt_emb = np.load(f'/tmp/analysis_{config["task"]}_emb_{config["run"]}_{model_criteria}.npy')
        # gt_labels = np.load(f'/tmp/analysis_{config["task"]}_labels_{config["run"]}_{model_criteria}.npy')
        # anatomy_embs = [np.load(f'/tmp/analysis_{config["task"]}_emb_{config["run"]}_{model_criteria}_{anatomy_name}.npy') for anatomy_name in anatomy_names]

        compute_occluded_metrics(
            config,
            model_criteria,
            gt_emb,
            gt_labels,
            anatomy_embs,
            nine_class_labels,
            anatomy_names,
            top_k=10,
            dist_metric='cosine'
        )
