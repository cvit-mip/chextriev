import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import faiss
from prettytable import PrettyTable

def calculate_metric(targets, skip=True):
    mAP, mHR, mRR = [], [], []
    consider_indices = []

    for consider_idx, target in enumerate(targets):
        if skip and target.sum() == 0:
            continue
        consider_indices.append(consider_idx)

        pos = 0
        found_hit = False
        ap = []
        for i, t in enumerate(target):
            if t:
                pos += 1
                ap.append(pos/(i+1))
                if not found_hit: 
                    mRR.append(1/(i+1))
                found_hit = True
        mHR.append(pos/len(target))
        if not found_hit:
            mAP.append(0)
            mRR.append(0)
        else:
            mAP.append(np.mean(ap))

    return mAP, mHR, mRR, consider_indices
    
    if len(mAP) == 0:
        return 0., 0., 0.

    return np.mean(mAP), np.mean(mHR), np.mean(mRR)


def compute_metrics(config, model_criteria, all_emb, all_labels, all_label_names, top_k=10, dist_metric='cosine', is_save=True, query_emb=None, placement=None, skip=True, naren=False, view=None, sex=None, age=None):

    # build the index using faiss
    d = all_emb.shape[1]
    if dist_metric == 'cosine':
        faiss_retriever = faiss.IndexFlatIP(d)
        all_emb = all_emb/np.linalg.norm(all_emb, axis=1, keepdims=True)
        query_emb = query_emb/np.linalg.norm(query_emb, axis=1, keepdims=True) if query_emb is not None else all_emb
    else:
        faiss_retriever = faiss.IndexFlatL2(d)
        query_emb = query_emb if query_emb is not None else all_emb
    faiss_retriever.add(all_emb)
    print(f'\nFaiss trained {faiss_retriever.is_trained} on {faiss_retriever.ntotal} vectors of size {faiss_retriever.d}')

    # targets contains the retrieved labels checked against ground truth labels
    # predictions contain the distances of the retrieved labels
    all_targets_for_metric = {lbl_name: [] for lbl_name in all_label_names}
    all_target_indices = {lbl_name: [] for lbl_name in all_label_names}
    all_query_indices = {lbl_name: [] for lbl_name in all_label_names}

    # keep count of number of queries per class for weighted average
    weights = np.zeros(len(all_label_names))

    # perform retrieval and save the input required for the metrics
    for _, (emb, query_labels) in enumerate(tqdm(zip(query_emb, all_labels), total=len(query_emb), desc=f'Retreiving top-{top_k} {dist_metric}...')):
        
        # expand dimension
        emb = emb[np.newaxis, ...]
        query_labels = query_labels[np.newaxis, ...]
        
        # perform retrieval
        D, I = faiss_retriever.search(emb, top_k+1)

        # find the corresponding labels from the retrieved indices
        # ignore the first one as it the query itself
        labels = all_labels[I[:, 1:]]

        # we only care about query labels that are present
        target = torch.tensor(labels == 1)

        # class wise metrics
        for i, label_name in enumerate(all_label_names):
            
            # works with batched retrieval as well
            consider_batches = query_labels[:, i] == 1
            if consider_batches.sum() == 0:
                continue
            # extract only the relevant batches
            temp_target = target[consider_batches]
            temp_target_indices = I[consider_batches, 1:]
            temp_query_indices = I[consider_batches, 0]

            # save necessary values
            all_targets_for_metric[label_name].append(temp_target[:, :, i])
            all_target_indices[label_name].append(temp_target_indices)
            all_query_indices[label_name].append(temp_query_indices)

            # update weights
            weights[i] += consider_batches.sum().item()

    # convert to tensors
    all_targets_for_metric = {k: torch.cat(v) if len(v) else None for k, v in all_targets_for_metric.items()}
    all_target_indices = {k: np.concatenate(v) if len(v) else None for k, v in all_target_indices.items()}
    all_query_indices = {k: np.concatenate(v) if len(v) else None for k, v in all_query_indices.items()}

    # dump the results to dataframe
    disease_dump = []
    target_hit_dump = []
    target_indices_dump = []
    query_indices_dump = []
    AP, HR, RR = [], [], []

    # for pretty tables
    t = PrettyTable(['Label Name', 'mAP', 'mHR', 'mRR'])
    print()

    # compute class wise metrics
    avg_values = []
    for i, label_name in enumerate(tqdm(all_label_names, desc='Computing metrics...')):

        if all_targets_for_metric[label_name] is None:
            new_ap, new_hr, new_rr = [], [], []
            new_map, new_mhr, new_mrr = 0., 0., 0.
        else:
            new_ap, new_hr, new_rr, consider_indices = calculate_metric(all_targets_for_metric[label_name], skip=skip)
            new_map, new_mhr, new_mrr = np.mean(new_ap), np.mean(new_hr), np.mean(new_rr)

            # update the dump lists
            consider_indices = set(consider_indices)
            disease_dump.extend([label_name]*len(new_ap))
            target_hit_dump.extend([val for idx, val in enumerate(all_targets_for_metric[label_name].tolist()) if idx in consider_indices])
            target_indices_dump.extend([val for idx, val in enumerate(all_target_indices[label_name].tolist()) if idx in consider_indices])
            query_indices_dump.extend([val for idx, val in enumerate(all_query_indices[label_name].tolist()) if idx in consider_indices])
            AP.extend(new_ap)
            HR.extend(new_hr)
            RR.extend(new_rr)
        
        avg_values.append([new_map, new_mhr, new_mrr])
        
        # add the row to the table
        t.add_row([label_name,  np.round(new_map, 3), np.round(new_mhr, 3), np.round(new_mrr, 3)])
    
    avg_map, avg_mhr, avg_mrr = np.mean(avg_values, axis=0)
    t.add_row(['Class Average', np.round(avg_map, 3), np.round(avg_mhr, 3), np.round(avg_mrr, 3)])
    
    # add the average row to the table and write to file
    # weights = np.load('data/mimic_cxr_jpg/test_weights.npy')
    avg_map, avg_mhr, avg_mrr = np.average(avg_values, axis=0, weights=weights)
    t.add_row(['Class Weighted Average', np.round(avg_map, 3), np.round(avg_mhr, 3), np.round(avg_mrr, 3)])
    
    print(t)
    if not is_save:
        return t, avg_map, avg_mhr, avg_mrr

    # create directory for the run
    dir_name = f'{config["task"]}/{config["run"]}' if not naren else f'naren/{config["task"]}/{config["run"]}'
    dir_name += '/occluded' if 'occluded_anatomies' in config else ''
    dir_name += f'/orig' if not skip else ''
    dir_name += f'/{placement}' if placement is not None else ''
    df_dump_dir = 'results/' + dir_name
    dir_name = 'output/' + dir_name
    os.makedirs(df_dump_dir, exist_ok=True)
    os.makedirs(dir_name, exist_ok=True)
    
    # save the table to file
    file_name = f'{len(all_label_names)}_classes_{model_criteria}{"" if config["is_global_feat"] else "_no_global_feat"}{"_concat_global_feat" if config["concat_global_feature"] else ""}{"_pruned" if config["prune"] else ""}_top_{top_k}_{dist_metric}'
    file_name += f'_{config["occluded_anatomies"]}' if 'occluded_anatomies' in config else ''
    file_name += f'_{view}' if view is not None else ''
    file_name += f'_{sex}' if sex is not None else ''
    file_name += f'_{age}' if age is not None else ''
    file_name += '.txt'
    df_dump_file_name = f'{df_dump_dir}/{file_name}'
    file_name = f'{dir_name}/{file_name}'
    with open(file_name, 'w') as f:
        f.write(str(t))

    # save the results to dataframe
    df = pd.DataFrame({
        'label_name': disease_dump,
        'target_hit': target_hit_dump,
        'target_indices': target_indices_dump,
        'query_indices': query_indices_dump,
        'AP': AP,
        'HR': HR,
        'RR': RR
    })
    df.to_csv(df_dump_file_name, index=False)


def compute_occluded_metrics(config, model_criteria, gt_emb, gt_labels, anatomy_embs, all_label_names, anatomy_names, top_k=10, dist_metric='cosine'):

    # build the index using faiss
    d = gt_emb.shape[1]
    if dist_metric == 'cosine':
        faiss_retriever = faiss.IndexFlatIP(d)
        gt_emb = gt_emb/np.linalg.norm(gt_emb, axis=1, keepdims=True)
        anatomy_embs = [anatomy_emb/np.linalg.norm(anatomy_emb, axis=1, keepdims=True) for anatomy_emb in anatomy_embs]
    else:
        faiss_retriever = faiss.IndexFlatL2(d)
    faiss_retriever.add(gt_emb)
    print(f'\nFaiss trained {faiss_retriever.is_trained} on {faiss_retriever.ntotal} vectors of size {faiss_retriever.d}')

    # create directory for the run
    dir_name = f'output/{config["task"]}/{config["run"]}'
    os.makedirs(dir_name, exist_ok=True)
    
    # save the table to file
    file_name = f'{dir_name}/Aanlysis_{len(all_label_names)}_classes_{model_criteria}{"_concat_global_feat" if config["concat_global_feature"] else ""}{"_pruned" if config["prune"] else ""}_top_{top_k}_{dist_metric}.txt'
    
    with open(file_name, 'w') as f:
        t, _, _, _ = compute_metrics(
            config, model_criteria, 
            gt_emb, gt_labels, all_label_names, 
            top_k=top_k, dist_metric=dist_metric, is_save=False)
        f.write(f'Results without occlusion:\n{str(t)}\n\n')
        print(t)

        for anatomy_occluded_emb, anatomy_name in zip(anatomy_embs, anatomy_names):
            t = compute_metrics(
                config, model_criteria, 
                anatomy_occluded_emb, gt_labels, all_label_names, 
                top_k=top_k, dist_metric=dist_metric, is_save=False)
            f.write(f'Results occluding {anatomy_name} node:\n{str(t)}\n\n')
            print(t)
    