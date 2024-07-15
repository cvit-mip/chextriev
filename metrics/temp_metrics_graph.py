import os.path
from tqdm import tqdm
import numpy as np
from glob import glob
import torch
import faiss
from torch.utils.data import DataLoader
import argparse

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

all_lables = [  'Atelectasis',
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

def get_metrics(query_label, labels):
    
    metrics = {f'{label_name}_{m}': 0 for label_name in all_lables for m in ['MAP', 'MHR', 'MRR']}
    for i, label_name in enumerate(all_lables):
    
        # initialise mean hit ratio, mean reciprocal rank, and mean average precision
        MHR, MRR, MAP = [], [], []
        
        # position, rank, and flag
        pos, mrr_flag = 0, False
        
        # iterate over the neighbors
        for rank, label in enumerate(labels):

            # its a hit
            if (query_label[i] == label[i]):
                pos += 1
                MHR.append(1)
                MAP.append(pos/(rank+1))

                # its the first hit
                if not mrr_flag:
                    MRR.append(pos/(rank+1))
                    mrr_flag = True
            
            # its a miss
            else:
                MHR.append(0)
                MAP.append(0)
        
        MRR = MRR[0] if len(MRR) else 0
        
        metrics[f'{label_name}_MAP'] = sum(MAP)/len(MAP)
        metrics[f'{label_name}_MHR'] = sum(MHR)/len(MHR)
        metrics[f'{label_name}_MRR'] = MRR
        # sum(MAP)/len(MAP), sum(MHR)/len(MHR), MRR

    return metrics


def print_metrics(file_name: str):
    with open(file_name, 'r') as f:
        saved_metrics = f.read().split('\n')

    MAP, MHR, MRR = [], [], []

    for metric in saved_metrics:
        if 'MAP' in metric:
            MAP.append(float(metric.split(': ')[1]))
        elif 'MHR' in metric:
            MHR.append(float(metric.split(': ')[1]))
        elif 'MRR' in metric:
            MRR.append(float(metric.split(': ')[1]))

    print(f'MAP: {np.mean(MAP):.3f}, MHR: {np.mean(MHR):.3f}, MRR: {np.mean(MRR):.3f}')


if __name__ == '__main__':

    config['data_dir'] ='/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
    device = 'cuda:3'
    # all_files = sorted(glob(f'{config["data_dir"]}/**/*.jpg', recursive=True))
    # print(len(all_files))

    d = 1024
    index = faiss.IndexFlatL2(d)   # build the index
    print(index.is_trained)

    if config['task'] == 'mimic-cxr-jpg':
        from model.mimic_cxr_jpg import CustomModel
        from dataloader.mimic_cxr_jpg import CustomDataset
    elif config['task'] == 'graph-jpg':
        from model.graph_jpg import CustomModel
        from dataloader.graph_jpg import CustomDataset

    # train_dataset = CustomDataset(config, split='train', to_gen=-1)
    val_dataset   = CustomDataset(config, split='val', to_gen=-1)

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=32, collate_fn=train_dataset.collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=val_dataset.collate_fn)

    # load model from checkpoint
    model = CustomModel.load_from_checkpoint('/home/ssd_scratch/users/arihanth.srikar/checkpoints/mimic-cxr/graph_jpg/epoch=6-last.ckpt', config=config)
    saved_model = True
    # model = CustomModel(config)
    model.fc = torch.nn.Identity()
    model = model.to(device)

    all_emb = []

    if not saved_model and os.path.exists('/home/ssd_scratch/users/arihanth.srikar/graph_emb.npy'):
        all_emb = np.load('/home/ssd_scratch/users/arihanth.srikar/graph_emb.npy')
        index.add(all_emb)
        assert len(all_emb) > 0, 'index not loaded properly'
    elif saved_model and os.path.exists('/home/ssd_scratch/users/arihanth.srikar/graph_emb_finetuned.npy'):
        all_emb = np.load('/home/ssd_scratch/users/arihanth.srikar/graph_emb_finetuned.npy')
        index.add(all_emb)
        assert len(all_emb) > 0, 'index not loaded properly'
    
    else:
        for i, batch in enumerate(tqdm(val_loader)):
            batch = [b.to(device) for b in batch]
            emb, _ = model.retrieval_pass(batch)
            emb = emb.detach().cpu().numpy()
            all_emb.append(emb)
            index.add(emb)

        all_emb = np.concatenate(all_emb)
        if not saved_model:
            np.save('/home/ssd_scratch/users/arihanth.srikar/graph_emb.npy', all_emb)
        else:
            np.save('/home/ssd_scratch/users/arihanth.srikar/graph_emb_finetuned.npy', all_emb)

    print(index.ntotal)

    metrics = {f'{label_name}_{m}': [] for label_name in all_lables for m in ['MAP', 'MHR', 'MRR']}

    with tqdm(val_loader) as pbar:
        for batch in pbar:
            batch = [b.to(device) for b in batch]
            emb, query_labels = model.retrieval_pass(batch)
            emb, query_labels = emb.detach().cpu().numpy(), query_labels.detach().cpu().numpy()
            D, I = index.search(emb, 5)

            labels = [[val_dataset.__getitem__(i)[-1] for i in I[j][1:]] for j in range(I.shape[0])]

            for query_label, target_label in zip(query_labels, labels):
                m = get_metrics(query_label, target_label)
                for k, v in m.items():
                    metrics[k].append(v)
            
            pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items()})

    f_name = config['file_name']
    with open(f_name, 'w') as f:
        for k, v in metrics.items():
            f.write(f'{k}: {sum(v)/len(v)}\n')

    print_metrics(f_name)
