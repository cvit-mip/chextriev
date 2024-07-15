import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from glob import glob
import torch
from torch.utils.data import DataLoader
import argparse
import faiss
from scipy.stats import ttest_ind
from prettytable import PrettyTable

from dataloader.anaxnet import OcclusionDataset

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


def calculate_metric(targets):
    mAP, mHR, mRR = [], [], []

    for _, target in enumerate(targets):
        if target.sum() == 0:
            mAP.append(0)
            mHR.append(0)
            mRR.append(0)
            continue

        pos = 0
        found_hit = False
        AP = []
        for i, t in enumerate(target):
            if t:
                pos += 1
                AP.append(pos/(i+1))
                if not found_hit: mRR.append(1/(i+1))
                found_hit = True
        mHR.append(int(found_hit))
        mAP.append(np.mean(AP) if len(AP) > 0 else 0)

    return mAP, mHR, mRR


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

    seed_everything(42)

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
    else:
        raise NotImplementedError
    
    store_dir = '/scratch/arihanth.srikar/dump_data'
    anatomy_names = [
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
    consider_anatomies = [[]] + [a_list for a_list in anatomy_names]
    
    try:
        model_criteria = 'auc'
        all_emb = np.load(f'{store_dir}/{config["run"]}/{config["task"]}_emb_{model_criteria}_original.npy')
        all_labels = np.load(f'{store_dir}/{config["run"]}/{config["task"]}_labels_{model_criteria}_original.npy')
        occluded_emb = {'-'.join(anatomy_list).replace(' ', '_') if len(anatomy_list) else 'original': np.load(f'{store_dir}/{config["run"]}/{config["task"]}_emb_{model_criteria}_{"-".join(anatomy_list).replace(" ", "_") if len(anatomy_list) else "original"}.npy') for anatomy_list in consider_anatomies}
        print('Loading embeddings and labels...')

    except:

        device = 'cuda'

        print("Loading dataset")
        df = pd.read_json('/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/mimic_coco_filtered.json')
        temp_df = pd.read_csv('data/mimic_cxr_jpg/mimic-cxr-2.0.0-final.csv')
        temp_df.rename(columns={'dicom_id': 'image_id'}, inplace=True)
        df = df.merge(temp_df, on='image_id', how='left')
        print("Dataset loaded")

        # load model from checkpoint
        model_paths = sorted(glob(f'/home/ssd_scratch/users/arihanth.srikar/checkpoints/mimic-cxr/{config["run"]}/*.ckpt'))
        
        for model_criteria in ['auc']:

            # load model from checkpoint
            try:
                model_path = [md_name for md_name in model_paths if model_criteria in md_name][-1]
                model = CustomModel.load_from_checkpoint(model_path, config=config)
                print(f'Loading best model based on {model_criteria}')
            except:
                if model_criteria != 'randomly_initialized': continue
                model = CustomModel(config)
                print(f'Loading {model_criteria} model')

            model = model.to(device)
            model = model.eval()

            for anatomy_list in consider_anatomies:
                config['occluded_anatomies'] = '-'.join(anatomy_list).replace(' ', '_') if len(anatomy_list) else 'original'
                test_dataset = OcclusionDataset(df, split='test', occlude_anatomy=anatomy_list)
                test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
                print(f'Occulding {anatomy_list} from the test set using {model_criteria} model')

                all_emb = []
                all_labels = []
                
                with torch.no_grad():
                    for i, batch in enumerate(tqdm(test_loader)):
                        node_feats = batch['node_feat'].to(device)
                        global_feat = batch['global_feat'].to(device) if config['is_global_feat'] else None
                        y = torch.sum(batch['y'], dim=1) > 0
                        emb = model.retrieval_pass(node_feats, global_feat, config['concat_global_feature'])
                        all_emb.append(emb.detach().cpu().numpy())
                        all_labels.append(y.numpy())

                # cache for later use
                all_emb = np.concatenate(all_emb)
                all_labels = np.concatenate(all_labels)
                
                os.makedirs(store_dir, exist_ok=True)
                os.makedirs(f'{store_dir}/{config["run"]}', exist_ok=True)
                np.save(f'{store_dir}/{config["run"]}/{config["task"]}_emb_{model_criteria}_{config["occluded_anatomies"]}.npy', all_emb)
                np.save(f'{store_dir}/{config["run"]}/{config["task"]}_labels_{model_criteria}_{config["occluded_anatomies"]}.npy', all_labels)
                print(f'Saved {config["occluded_anatomies"]} embeddings and labels')

    try:
        df = pd.read_pickle(f'{store_dir}/{config["run"]}_dump.pickle')
        print('Loaded dump file')
    
    except:
        # occluded_emb['original'] = all_emb
        df = pd.DataFrame(columns=['occluded_anatomy', 'disease', 'AP', 'HR', 'RR', 'retrieved_match', 'retrieved_similarity'])

        # normalise
        occluded_emb = {k: v/np.linalg.norm(v, axis=1, keepdims=True) for k, v in occluded_emb.items()}
        query_emb = occluded_emb["original"]
        
        # for the occlusion of each anatomy
        # run retrieval on all diseases
        for k_anatomy, v_anatomy in occluded_emb.items():

            # build the index using faiss
            top_k = 10
            d = all_emb.shape[1]
            faiss_retriever = faiss.IndexFlatIP(d)
            faiss_retriever.add(v_anatomy)
            print(f'\nFaiss trained {faiss_retriever.is_trained} on {faiss_retriever.ntotal} vectors of size {faiss_retriever.d}')

            # targets contains the retrieved labels checked against ground truth labels
            # predictions contain the distances of the retrieved labels
            all_targets_for_metric = {lbl_name: [] for lbl_name in nine_class_labels}
            all_similarity_for_metric = {lbl_name: [] for lbl_name in nine_class_labels}

            # perform retrieval and save the input required for the metrics
            for _, (emb, query_labels) in enumerate(tqdm(zip(query_emb, all_labels), total=len(query_emb), desc=f'Retreiving top-{top_k} {k_anatomy}...')):
                
                # expand dimension
                emb = emb[np.newaxis, ...]
                query_labels = query_labels[np.newaxis, ...]
                
                # perform retrieval
                D, I = faiss_retriever.search(emb, top_k+1) if k_anatomy == "original" else faiss_retriever.search(emb, top_k)

                # find the corresponding labels from the retrieved indices
                # ignore the first one as it the query itself
                labels = all_labels[I[:, 1:]] if k_anatomy == "original" else all_labels[I]
                similarity = torch.tensor(D[:, 1:]) if k_anatomy == "original" else torch.tensor(D)

                # we only care about query labels that are present
                target = torch.tensor(labels == 1)

                # class wise metrics
                for i, label_name in enumerate(nine_class_labels):
                    
                    # works with batched retrieval as well
                    consider_batches = query_labels[:, i] == 1
                    if consider_batches.sum() == 0:
                        continue
                    # extract only the relevant batches
                    temp_target = target[consider_batches]
                    temp_similarity = similarity[consider_batches]

                    # save necessary values
                    all_targets_for_metric[label_name].append(temp_target[:, :, i])
                    all_similarity_for_metric[label_name].append(temp_similarity)

            # convert to tensors
            all_targets_for_metric = {k: torch.cat(v) for k, v in all_targets_for_metric.items()}
            all_similarity_for_metric = {k: torch.cat(v) for k, v in all_similarity_for_metric.items()}

            # calculate metrics
            for k_diseases, v_diseases in all_targets_for_metric.items():
                mAP, mHR, mRR = calculate_metric(v_diseases)

                # save the results
                cur_anatomy = [k_anatomy] * len(all_targets_for_metric[k_diseases])
                cur_disease = [k_diseases] * len(all_targets_for_metric[k_diseases])
                cur_df = pd.DataFrame({
                    'occluded_anatomy': cur_anatomy, 
                    'disease': cur_disease, 
                    'AP': mAP, 'HR': mHR, 'RR': mRR, 
                    'retrieved_match': all_targets_for_metric[k_diseases].tolist(), 
                    'retrieved_similarity': all_similarity_for_metric[k_diseases].tolist(),
                    })
                df = pd.concat([df, cur_df], axis=0)

        # write to file
        df.to_pickle(f'{store_dir}/{config["run"]}_dump.pickle')
        print('Saved dump file')

    t = PrettyTable(['Anatomy', 'Disease', 'p_stat_0.05', 'p_stat_0.01', 'p_val'])
    # calculate stats for each anatomy
    for d_name in nine_class_labels:
        for a_name in anatomy_names:
            a_name = '-'.join(a_name).replace(' ', '_') if len(a_name) else 'original'
            t_stat, p_val = ttest_ind(df[(df['disease'] == d_name) & (df['occluded_anatomy'] == a_name)]['AP'], df[(df['disease'] == d_name) & (df['occluded_anatomy'] == 'original')]['AP'])
            p_005 = 'T' if p_val <= 0.05 else '-'
            p_001 = 'T' if p_val <= 0.01 else '-'
            t.add_row([a_name, d_name, p_005, p_001, f'{p_val:.3f}'])
        t.add_row(['', '', '', '', ''])
    print(t)
