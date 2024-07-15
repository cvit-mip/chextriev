import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
import argparse

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

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
    'pneumonia']

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


if __name__ == '__main__':

    config['task'] = 'generalized-image-embeddings'
    config['run'] = 'pretrained'
    config['data_dir'] = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0/files/'
    device = 'cuda'

    all_file_names = glob(f'{config["data_dir"]}**/*.tfrecord', recursive=True)

    print("Loading dataset")
    try:
        df = pd.read_json('/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/mimic_coco_filtered.json')
    except:
        df = pd.read_json('/scratch/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/mimic_coco_filtered.json')
    temp_df = pd.read_csv('data/mimic_cxr_jpg/mimic-cxr-2.0.0-final.csv')
    temp_df.rename(columns={'dicom_id': 'image_id'}, inplace=True)
    df = df.merge(temp_df, on='image_id', how='left')
    print("Dataset loaded")

    long_prefix = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0/files'
    test_file_names = df[df['split'] == 'test'].apply(lambda x: f"{long_prefix}/p{str(x['subject_id'])[:2]}/p{int(x['subject_id'])}/s{int(x['study_id'])}/{x['image_id']}.tfrecord", axis=1).to_list()

    test_dataset = tf.data.TFRecordDataset(test_file_names)

    all_emb = []
    all_labels = []
    positioned_labels = []

    for i, test in enumerate(tqdm(test_dataset.take(-1), total=len(test_file_names))):
        example = tf.train.Example()
        example.ParseFromString(test.numpy())
        
        emb = np.array(example.features.feature['embedding'].float_list.value).astype(np.float32).reshape(1, -1)
        
        image_id = example.features.feature['image/id'].bytes_list.value[0].decode('utf-8')
        pid = image_id.split('/')[7]
        sid = image_id.split('/')[8]
        did = image_id.split('/')[9].split('.')[0]
        
        y_9 = []
        sample_data = df[df['image_id'] == did].to_dict(orient='records')[0]
        for annotation in sorted(sample_data['annotations'], key=lambda k: k['category_id']):
            y_9.append(annotation['attributes'])
        y_9 = np.array(y_9)
        y = np.sum(y_9, axis=0) > 0

        all_emb.append(emb)
        all_labels.append(y[np.newaxis, ...])

        # modified right, left, middle, and all structure labels
        only_right = np.sum(y_9[:7], axis=0) > 0
        only_left = np.sum(y_9[7:14], axis=0) > 0
        only_middle = np.sum(y_9[14:], axis=0) > 0
        all_structures = np.sum(y_9, axis=0) > 0
        
        consider_indices = [thirtysix_class_labels.index(lbl_name) for lbl_name in consider_class_labels]
        consider_labels = np.concatenate([only_right, only_left, only_middle, all_structures], axis=0)[consider_indices]
        positioned_labels.append(consider_labels[np.newaxis, ...])

    all_emb = np.concatenate(all_emb)
    all_labels = np.concatenate(all_labels)
    positioned_labels = np.concatenate(positioned_labels)

    np.save(f'/tmp/{config["run"]}_emb.npy', all_emb)
    np.save(f'/tmp/{config["run"]}_labels.npy', all_labels)
    np.save(f'/tmp/{config["run"]}_positioned_labels.npy', positioned_labels)
    
    assert all_emb.shape[0] == all_labels.shape[0], 'Number of embeddings and labels do not match'

    for top_k in [3, 5, 10]:
        # compute metrics
        compute_metrics(config, '', all_emb, all_labels, nine_class_labels, top_k=top_k, dist_metric='cosine', skip=True)
        compute_metrics(config, '', all_emb, all_labels, nine_class_labels, top_k=top_k, dist_metric='cosine', skip=False)

        # right, left, and middle structure disease metrics
        compute_metrics(config, '', all_emb, positioned_labels, consider_class_labels, top_k=top_k, dist_metric='cosine', skip=True)
        compute_metrics(config, '', all_emb, positioned_labels, consider_class_labels, top_k=top_k, dist_metric='cosine', skip=False)
