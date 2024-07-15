import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from glob import glob
from prettytable import PrettyTable
import torch
import faiss
from torch.utils.data import DataLoader
import argparse
from torchmetrics.retrieval import RetrievalMAP, RetrievalMRR, RetrievalHitRate

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from torchvision.models import densenet121, DenseNet121_Weights
from dataloader.anaxnet import CustomDataset

parser = argparse.ArgumentParser()

parser.add_argument('--emb_dim', type=int, default=512, help='embedding dimension')
parser.add_argument('--edge_dim', type=int, default=32, help='edge embedding dimension')
parser.add_argument('--num_classes', type=int, default=14, help='number of classes')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--grad_accum', type=int, default=1, help='gradient accumulation steps')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers')

parser.add_argument('--task', type=str, default='mimic-cxr-emb', help='model name')
parser.add_argument('--save_dir', type=str, default='/home/ssd_scratch/users/arihanth.srikar/checkpoints', help='save directory')
parser.add_argument('--entity', type=str, default='arihanth', help='wandb entity name')
parser.add_argument('--project', type=str, default='mimic-cxr', help='wandb project name')
parser.add_argument('--run', type=str, default='test', help='wandb run name')

parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0], help='list of gpu ids')
parser.add_argument('--set_precision', action=argparse.BooleanOptionalAction, help='set precision')
parser.add_argument('--log', action=argparse.BooleanOptionalAction, help='log to wandb')
parser.add_argument('--compile', action=argparse.BooleanOptionalAction, help='compile model')
parser.add_argument('--num_workers', type=int, default=10, help='number of workers')
parser.add_argument('--train', action=argparse.BooleanOptionalAction, help='train model')

parser.add_argument('--validate_every', type=int, default=1000, help='train for n epochs')
parser.add_argument('--validate_for', type=int, default=200, help='validate for n epochs')

config = vars(parser.parse_args())

all_label_names = [
            'lung opacity', 
            'pleural effusion', 
            'atelectasis', 
            'enlarged cardiac silhouette',
            'pulmonary edema/hazy opacity', 
            'pneumothorax', 
            'consolidation', 
            'fluid overload/heart failure', 
            'pneumonia']

os.environ['TORCH_HOME'] = '/ssd_scratch/users/arihanth.srikar'

def calculate_metric(targets):
    mAP, mHR, mRR = [], [], []

    for _, target in enumerate(targets):
        if target.sum() == 0:
            mHR.append(0)
            continue

        pos = 0
        found_hit = False
        for i, t in enumerate(target):
            if t:
                pos += 1
                mAP.append(pos/(i+1))
                mRR.append(1/pos)
                found_hit = True
        mHR.append(int(found_hit))

    if len(mAP) == 0:
        return 0., 0., 0.

    return np.mean(mAP), np.mean(mHR), np.mean(mRR)

if __name__ == '__main__':
    
    device = 'cuda'

    print("Loading dataset")
    df = pd.read_json('/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/mimic_coco_filtered.json')
    temp_df = pd.read_csv('data/mimic_cxr_jpg/mimic-cxr-2.0.0-final.csv')
    temp_df.rename(columns={'dicom_id': 'image_id'}, inplace=True)
    df = df.merge(temp_df, on='image_id', how='left')
    print("Dataset loaded")

    test_dataset = CustomDataset(df, split='test')
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    
    model_criteria = 'pretrained'
    model = densenet121(DenseNet121_Weights.DEFAULT)
    model.classifier = torch.nn.Identity()
    model = model.to(device)
    model = model.eval()

    # store embeddings and labels
    all_emb = []
    all_labels = []

    # extract embeddings
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            x = batch['global_feat'].to(device)
            y = torch.sum(batch['y'], dim=1) > 0
            emb = model(x).detach().cpu().numpy()
            all_emb.append(emb)
            all_labels.append(y.numpy())

    # cache for later use
    all_emb = np.concatenate(all_emb)
    all_labels = np.concatenate(all_labels)
    np.save('/tmp/nine_class_emb.npy', all_emb)
    np.save('/tmp/nine_class_labels.npy', all_labels)

    print('Loading embeddings and labels...')
    all_emb = np.load('/tmp/nine_class_emb.npy')
    all_labels = np.load('/tmp/nine_class_labels.npy')
    print(all_emb.shape, all_labels.shape)

    # build the index using faiss
    d = all_emb.shape[1]
    faiss_retriever = faiss.IndexFlatL2(d)
    faiss_retriever.add(all_emb)
    print(f'\nFaiss trained {faiss_retriever.is_trained} on {faiss_retriever.ntotal} vectors of size {faiss_retriever.d}')

    # targets contains the retrieved labels checked against ground truth labels
    # predictions contain the distances of the retrieved labels
    all_targets_for_metric = {lbl_name: [] for lbl_name in all_label_names}
    all_predicted_for_metric = {lbl_name: [] for lbl_name in all_label_names}

    # perform retrieval and save the input required for the metrics
    for _, (emb, query_labels) in enumerate(tqdm(zip(all_emb, all_labels), total=len(all_emb), desc='Retreiving...')):
        
        # expand dimension
        emb = emb[np.newaxis, ...]
        query_labels = query_labels[np.newaxis, ...]
        
        # perform retrieval
        D, I = faiss_retriever.search(emb, 11)

        # find the corresponding labels from the retrieved indices
        # ignore the first one as it the query itself
        labels = all_labels[I[:, 1:]]
        distances = torch.softmax(torch.tensor(D[:, 1:]), dim=1)

        # we only care about query labels that are present
        target = torch.tensor(labels == 1)
        predicted = 1/(distances+1)

        # class wise metrics
        for i, label_name in enumerate(all_label_names):
            
            # works with batched retrieval as well
            consider_batches = query_labels[:, i] == 1
            if consider_batches.sum() == 0:
                continue
            # extract only the relevant batches
            temp_target = target[consider_batches]
            temp_predicted = predicted[consider_batches]

            # save necessary values
            all_targets_for_metric[label_name].append(temp_target[:, :, i])
            all_predicted_for_metric[label_name].append(temp_predicted)

    # convert to tensors
    all_targets_for_metric = {k: torch.cat(v) for k, v in all_targets_for_metric.items()}
    all_predicted_for_metric = {k: torch.cat(v) for k, v in all_predicted_for_metric.items()}
    all_indexes_for_metric = {k: torch.tensor([[i for _ in range(v.shape[1])] for i in range(v.shape[0])]) for k, v in all_targets_for_metric.items()}

    # for pretty tables
    t = PrettyTable(['Label Name', 'mAP', 'mHR', 'mRR'])
    print()

    # compute class wise metrics
    avg_map, avg_mhr, avg_mrr = 0, 0, 0
    for i, label_name in enumerate(tqdm(all_label_names, desc='Computing metrics...')):

        # # compute metrics
        # new_map = map_calc(all_predicted_for_metric[label_name], all_targets_for_metric[label_name], indexes=all_indexes_for_metric[label_name]).item()
        # new_mhr = mhr_calc(all_predicted_for_metric[label_name], all_targets_for_metric[label_name], indexes=all_indexes_for_metric[label_name]).item()
        # new_mrr = mrr_calc(all_predicted_for_metric[label_name], all_targets_for_metric[label_name], indexes=all_indexes_for_metric[label_name]).item()

        new_map, new_mhr, new_mrr = calculate_metric(all_targets_for_metric[label_name])
        
        # compute average across classes
        avg_map += (new_map/len(all_label_names))
        avg_mhr += (new_mhr/len(all_label_names))
        avg_mrr += (new_mrr/len(all_label_names))
        
        # add the row to the table
        t.add_row([label_name,  np.round(new_map, 3), np.round(new_mhr, 3), np.round(new_mrr, 3)])
    
    # add the average row to the table and write to file
    t.add_row(['Class Average', np.round(avg_map, 3), np.round(avg_mhr, 3), np.round(avg_mrr, 3)])
    
    print(t)

    # save the table to file
    file_name = f'output/densenet_9_classes_{model_criteria}.txt'
    with open(file_name, 'w') as f:
        f.write(str(t))
    