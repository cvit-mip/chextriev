import warnings
warnings.filterwarnings("ignore")

import os
import gc
import math
import pandas as pd
import argparse
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner.tuning import Tuner


parser = argparse.ArgumentParser()

parser.add_argument('--emb_dim', type=int, default=1024, help='embedding dimension')
parser.add_argument('--edge_dim', type=int, default=32, help='edge embedding dimension')
parser.add_argument('--num_classes', type=int, default=14, help='number of classes')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--grad_accum', type=int, default=1, help='gradient accumulation steps')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
parser.add_argument('--edge_index', type=int, default=128, help='edge index')
parser.add_argument('--graph_importance', type=float, default=1.0, help='graph importance bewteen 0 and 1')
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
    
    # clear gpu
    torch.cuda.empty_cache()
    gc.collect()
    
    # use this to set the precision of matmul operations
    torch.set_float32_matmul_precision('medium')
    seed_everything(42)

    # imagenet on chest xray dataset
    if config['task'] == 'mimic-cxr-jpg':
        from dataloader.image_only import CustomDataset
        from model.mimic_cxr_jpg import CustomModel
        config['data_dir'] = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'

    # fast loader on chest xray dataset
    elif 'fast' in config['task']:
        from dataloader.fast_loader import CustomDataset
        config['data_dir'] = '/scratch/arihanth.srikar/'
    
    # imagenet on chest xray dataset preprocessed and dumped as npy
    elif config['task'] == 'mimic-cxr-jpg-preprocessed':
        from dataloader.mimic_cxr_preprocessed import CustomDataset
        from model.mimic_cxr_jpg import CustomModel
        config['data_dir'] = '/scratch/arihanth.srikar/'
    
    # graph on chest xray dataset
    elif config['task'] == 'graph-jpg':
        from dataloader.graph_jpg import CustomDataset
        from model.graph_jpg import CustomModel
        config['data_dir'] = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'

    # use predefined embeddings for mimic chest xray dataset
    elif config['task'] == 'mimic-cxr-emb':
        from dataloader.mimic_cxr_emb import CustomDataset
        from model.mimic_cxr_emb import CustomModel
        config['data_dir'] = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0/files'

    elif config['task'] == 'gcn':
        from dataloader.gcn import CustomDataset
        from model.gcn import CustomModel
        config['data_dir'] = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/scene_tabular/'
    
    elif config['task'] == 'gcn_preprocessed':
        from dataloader.gcn_preprocessed import GCNPreprocessed
        from model.gcn_preprocessed import CustomModel
        config['data_dir'] = '/scratch/arihanth.srikar'
   
    elif config['task'] == 'gcn_attn':
        from dataloader.gcn_preprocessed import GCNPreprocessed
        from model.gcn_attn import CustomModel
        config['data_dir'] = '/scratch/arihanth.srikar'
    
    elif config['task'] == 'gcn_global':
        from dataloader.gcn_global import GCNPreprocessed
        from model.gcn_global import CustomModel
        config['data_dir'] = '/scratch/arihanth.srikar'
    
    elif config['task'] in ['anaxnet', 'anaxnet_custom', 'anaxnet_attn', 'anaxnet_attn_multilayer', 'densenet_9_classes', 
                            'densenet121' , 'resnet50', 'chexrelnet', 'contrastive', 'ath', 'xfactor', 'graph_benchmark', 'graph_transformer', 'vanilla_transformer',
                            'local_features']:
        from dataloader.anaxnet import CustomDataset
        if config['task'] == 'anaxnet':
            from model.anaxnet import CustomModel
        elif config['task'] == 'anaxnet_custom':
            from model.anaxnet_custom import CustomModel
        elif config['task'] == 'anaxnet_attn':
            from model.anaxnet_attn import CustomModel
        elif config['task'] == 'anaxnet_attn_multilayer':
            from model.anaxnet_attn_multilayer import CustomModel
        elif config['task'] in ['densenet_9_classes', 'densenet121', 'resnet50']:
            from model.nine_class_classifier import CustomModel
        elif config['task'] == 'chexrelnet':
            from model.chexrelnet import CustomModel
        elif config['task'] == 'contrastive':
            from model.contrastive import CustomModel
        elif config['task'] == 'ath':
            from model.ath import CustomModel
        elif config['task'] == 'xfactor':
            from model.xfactor import CustomModel
        elif config['task'] == 'graph_benchmark':
            from model.graph_benchmark import CustomModel
        elif config['task'] == 'graph_transformer' or  config['task'] == 'local_features':
            from model.graph_transformer import CustomModel
        elif config['task'] == 'vanilla_transformer':
            from model.vanilla_transformer import CustomModel
        
        print("Loading dataset")
        try:
            df = pd.read_json('/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/mimic_coco_filtered.json')
        except:
            df = pd.read_json('/scratch/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/mimic_coco_filtered.json')
        temp_df = pd.read_csv('data/mimic_cxr_jpg/mimic-cxr-2.0.0-final.csv')
        temp_df.rename(columns={'dicom_id': 'image_id'}, inplace=True)
        df = df.merge(temp_df, on='image_id', how='left')
        print("Dataset loaded")
    
    else:
        raise NotImplementedError
    
    # get number of cpus
    config['num_cpus'] = len(os.sched_getaffinity(0))
    config['num_gpus'] = torch.cuda.device_count()
    config['num_workers'] = min(config['num_workers'], config['num_cpus']//config['num_gpus'])
    print(f"Using {config['num_workers']} workers")
    
    if config['task'] == 'gcn':
        from torch_geometric.data import DataLoader

        train_dataset = CustomDataset(config, split='train')
        val_dataset   = CustomDataset(config, split='val')
        
        train_loader  = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=False, num_workers=config["num_workers"])
        val_loader    = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=False, num_workers=config["num_workers"])
        
        model = CustomModel(config)
    
    elif config['task'] in ['gcn_preprocessed', 'gcn_attn', 'gcn_global']:
        from torch_geometric.loader import DataLoader

        train_dataset = GCNPreprocessed(config, split='train', threshold=0.5)
        val_dataset   = GCNPreprocessed(config, split='val', threshold=0.5)
        test_dataset  = GCNPreprocessed(config, split='test', threshold=0.5)
        assert len(train_dataset) > 0 and len(val_dataset) > 0, "Dataset is empty"
        
        train_loader  = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config["num_workers"])
        val_loader    = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config["num_workers"])
        test_loader   = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config["num_workers"])
        
        model = CustomModel(config, num_nodes=18)
    
    elif config['task'] in ['mimic-cxr-jpg-preprocessed', 'mimic-cxr-jpg']:
        from torch.utils.data import DataLoader
        
        # initialize dataset
        train_dataset = CustomDataset(config, split='train')
        val_dataset   = CustomDataset(config, split='val')
        test_dataset  = CustomDataset(config, split='test')
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=False, num_workers=config["num_workers"])
        val_loader   = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=False, num_workers=config["num_workers"])
        test_loader  = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=False, num_workers=config["num_workers"])

        # initialize model
        model = CustomModel(config)

    elif config['task'] in ['anaxnet', 'anaxnet_custom', 'anaxnet_attn', 'anaxnet_attn_multilayer', 'densenet_9_classes', 
                            'densenet121', 'resnet50', 'chexrelnet', 'contrastive', 'ath', 'xfactor', 'graph_benchmark', 
                            'graph_transformer', 'vanilla_transformer', 'local_features']:
        from torch.utils.data import DataLoader
        if config['task'] == 'chexrelnet':
            from dataloader.chexrelnet import CustomDataset
            from torch_geometric.loader import DataLoader
        elif config['task'] == 'contrastive':
            from dataloader.contrastive import CustomDataset
        elif config['task'] == 'ath':
            from dataloader.ath import CustomDataset
        elif config['task'] == 'local_features':
            from dataloader.anaxnet import LocalFeatures as CustomDataset
        
        # initialize dataset
        train_dataset = CustomDataset(df, split='train', return_masked_img=config['contrastive'])
        val_dataset   = CustomDataset(df, split='val', return_masked_img=config['contrastive'])
        test_dataset  = CustomDataset(df, split='test', return_masked_img=config['contrastive'])
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=False, num_workers=config["num_workers"])
        val_loader   = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=False, num_workers=config["num_workers"])
        test_loader  = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=False, num_workers=config["num_workers"])

        batches_per_gpu = math.ceil(len(train_loader) / len(config['gpu_ids']))
        train_steps = math.ceil(batches_per_gpu / config['grad_accum']) * config['epochs']
        config['num_steps'] = train_steps

        # initialize model
        model = CustomModel(config, num_classes=config['num_classes'], num_nodes=config['num_nodes'])

    elif 'fast' in config['task']:
        from torch.utils.data import DataLoader
        
        # initialize dataset
        train_dataset = CustomDataset(split='train')
        val_dataset   = CustomDataset(split='val')
        test_dataset  = CustomDataset(split='test')
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=config["num_workers"])
        val_loader   = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True, num_workers=config["num_workers"])
        test_loader  = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True, num_workers=config["num_workers"])

        if config['task'] == 'anaxnet_attn_multilayer_fast':
            from model.anaxnet_attn_multilayer_fast import CustomModel

        # initialize model
        model = CustomModel(config)
    
    else:
        from torch.utils.data import DataLoader
        
        # initialize dataset
        train_dataset = CustomDataset(config, split='train', to_gen=config['batch_size']*config['validate_every'])
        val_dataset   = CustomDataset(config, split='val', to_gen=config['batch_size']*config['validate_for'])
        test_dataset  = CustomDataset(config, split='test', to_gen=config['batch_size']*config['validate_for'])
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=False, num_workers=config["num_workers"], collate_fn=train_dataset.collate_fn)
        val_loader   = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=False, num_workers=config["num_workers"], collate_fn=val_dataset.collate_fn)
        test_loader  = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=False, num_workers=config["num_workers"], collate_fn=test_dataset.collate_fn)

        # initialize model
        model = CustomModel(config)

    # initialize logger
    logger = WandbLogger(
        # entity=config['entity'],
        project=config['project'],
        name=config['run'],
        save_dir=config['save_dir'],
        mode='disabled' if not config['log'] else 'online',
        )
    
    # initialize checkpoint callback
    loss_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
        dirpath=f"{config['save_dir']}/{config['project']}/{config['run']}",
        filename="model-{epoch:02d}-{val_loss:.5f}",
        )
    loss_callback.CHECKPOINT_NAME_LAST = "{epoch:02d}-last"
    
    auc_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_auc",
        mode="max",
        dirpath=f"{config['save_dir']}/{config['project']}/{config['run']}",
        filename="model-{epoch:02d}-{val_auc:.5f}",
        )
    
    mAP_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_mAP",
        mode="max",
        dirpath=f"{config['save_dir']}/{config['project']}/{config['run']}",
        filename="model-{epoch:02d}-{val_mAP:.5f}",
        )
    
    early_stopping = EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=4,
    )
    
    save_model_callbacks = [loss_callback, auc_callback, early_stopping]
    save_model_callbacks += [mAP_callback] if 'contrastive' in config['task'] or 'transformer' in config['task'] or config['task'] == 'anaxnet_attn_multilayer' else []

    # initialize trainer
    trainer = pl.Trainer(
        # accelerator='gpu', devices=-1, strategy='auto',
        accelerator='gpu', devices=config['gpu_ids'], strategy='ddp_find_unused_parameters_true',
        max_epochs=config['epochs'], logger=logger,
        precision='bf16-mixed' if config['set_precision'] else '32-true',
        gradient_clip_val=0.5, gradient_clip_algorithm='norm',
        accumulate_grad_batches=config['grad_accum'],
        callbacks=save_model_callbacks,
        num_sanity_val_steps=0,
        val_check_interval=0.5,
        enable_progress_bar=True,
        )
    
    # # find best lr
    # lr_trainer = Tuner(trainer)
    # lr_finder = lr_trainer.lr_find(model, train_loader, val_loader)
    # new_lr = lr_finder.suggestion()
    # print(f"New learning rate: {new_lr}")
    # exit()
    
    if config['train']:
        # train model
        trainer.fit(model, train_loader, val_loader)
        trainer.test(model, dataloaders=test_loader)
    else:

        from glob import glob
        
        try:
            model_paths = sorted(glob(f'/home/ssd_scratch/users/arihanth.srikar/checkpoints/mimic-cxr/{config["task"]}/*.ckpt'))
            model_criteria = 'auc'
            model_path = [md_name for md_name in model_paths if model_criteria in md_name][-1]
        except:
            model_path = None
        
        trainer.test(model, test_loader, ckpt_path=model_path)
