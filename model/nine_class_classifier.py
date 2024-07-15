import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121, DenseNet121_Weights

from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryAUROC, MultilabelAUROC
from common_metrics import compute_metrics as retrieval_metrics

os.environ['TORCH_HOME'] = '/ssd_scratch/users/arihanth.srikar'


class CustomModel(pl.LightningModule):
    def __init__(self, config: dict, num_classes: int=9, num_nodes: int=18):
        super().__init__()
        self.config = config
        self.lr = config['lr']
        self.test_outputs = []

        if 'resnet' in config['task']:
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model.fc = nn.Identity()
            self.fc1 = nn.Linear(2048, 1024)
            if config["prune"]:
                self.model.layer4 = nn.Identity()
                self.fc1 = nn.Linear(1024, 1024)
        
        elif 'densenet' in config['task']:
            self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
            self.model.classifier = nn.Identity()
            self.fc1 = nn.Linear(1024, 1024)
        
        else:
            raise NotImplementedError(f'Unknown model {config["task"]}')
        
        # freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)

        self.save_hyperparameters()

        # self.model = torch.compile(self.model, disable=not config['compile'])
        self.calc_metrics = {
            'f1': BinaryF1Score(),
            'auc': BinaryAUROC(),
            'acc': BinaryAccuracy(),
            'recall': BinaryRecall(),
            'precision': BinaryPrecision(),
            'multi_auc': MultilabelAUROC(num_labels=config['num_classes']),
        }
        self.labels = [
            'lung opacity', 
            'pleural effusion', 
            'atelectasis', 
            'enlarged cardiac silhouette',
            'pulmonary edema/hazy opacity', 
            'pneumothorax', 
            'consolidation', 
            'fluid overload/heart failure', 
            'pneumonia']
        
        self.retrieval_metrics = retrieval_metrics

        self.val_epoch_end_outputs = []
        self.test_epoch_end_outputs = []
    
    def compute_metrics(self, logits, y):
        logits = logits.detach().cpu()
        y = y.detach().cpu().int()
        return {
            'f1': self.calc_metrics['f1'](logits, y),
            'auc': self.calc_metrics['auc'](logits, y),
            'acc': self.calc_metrics['acc'](logits, y),
            'recall': self.calc_metrics['recall'](logits, y),
            'precision': self.calc_metrics['precision'](logits, y),
        }
    
    def compute_loss(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits, y)
    
    def forward(self, x, y):
        if len(x.shape) == 5:
            B, N, C, H, W = x.shape
            x = x.reshape(B*N, C, H, W)
            emb = self.model(x)
            emb = emb.reshape(B, N, -1)
            emb = self.fc1(emb)
            ret_emb = torch.mean(emb, dim=1)
        else:
            emb = self.model(x)
            emb = self.fc1(emb)
            ret_emb = emb
        
        logits = self.relu(emb)
        logits = self.fc2(logits)
        loss = self.compute_loss(logits, y)
        return logits, loss, ret_emb
    
    def training_step(self, batch, batch_idx):
        x = batch['global_feat'] if self.config['graph_importance'] == 1 else batch['node_feat']
        y = (torch.sum(batch['y'], dim=1) > 0).float() if self.config['graph_importance'] else batch['y']
        logits, loss, emb = self(x, y)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['global_feat'] if self.config['graph_importance'] == 1 else batch['node_feat']
        y = (torch.sum(batch['y'], dim=1) > 0).float() if self.config['graph_importance'] else batch['y']
        logits, loss, emb = self(x, y)

        self.val_epoch_end_outputs.append((logits.detach().cpu(), y.detach().cpu(), loss.detach().cpu(), emb.detach().cpu()))

        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch['global_feat'] if self.config['graph_importance'] == 1 else batch['node_feat']
        y = (torch.sum(batch['y'], dim=1) > 0).float() if self.config['graph_importance'] else batch['y']
        logits, loss, emb = self(x, y)

        self.test_epoch_end_outputs.append((logits.detach().cpu(), y.detach().cpu(), loss.detach().cpu(), emb.detach().cpu()))

        return loss
    
    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:

        logits, ys, losses, emb = zip(*self.val_epoch_end_outputs)
        logits = torch.cat(logits)
        y = torch.cat(ys)
        loss = torch.mean(torch.tensor(list(losses)))
        emb = torch.cat(emb)

        self.common_end_epoch_function(logits, y, loss, emb, 'val')
    
        self.val_epoch_end_outputs = []
    
    @torch.no_grad()
    def on_test_epoch_end(self):

        logits, ys, losses, emb = zip(*self.test_epoch_end_outputs)
        logits = torch.cat(logits)
        y = torch.cat(ys)
        loss = torch.mean(torch.tensor(list(losses)))
        emb = torch.cat(emb)

        self.common_end_epoch_function(logits, y, loss, emb, 'test')
    
        self.test_epoch_end_outputs = []
    
    
    @torch.no_grad()
    def common_end_epoch_function(self, logits, y, loss, emb, split):

        average_metrics = {
            'acc': [],
            'auc': [],
            'f1': [],
            'precision': [],
            'recall': [],
        }

        # retrieval
        y_ret = torch.sum(y, dim=1) > 0 if len(y.shape) == 3 else y
        _, mAP, mHR, mRR = self.retrieval_metrics(self.config, split, emb.numpy(), y_ret.numpy(), 
                                                  self.labels, top_k=10, dist_metric='cosine', is_save=False)
        self.log(f'{split}_mAP', mAP, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{split}_mHR', mHR, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{split}_mRR', mRR, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # iterate over all labels
        for i, label_name in enumerate(self.labels):
            metrics = self.compute_metrics(logits[:, i], y[:, i]) if len(y.shape) == 2 else self.compute_metrics(logits[:, :, i], y[:, :, i])

            self.log(f'{split}_{label_name}_acc', metrics['acc'].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'{split}_{label_name}_auc', metrics['auc'].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'{split}_{label_name}_f1', metrics['f1'].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'{split}_{label_name}_precision', metrics['precision'].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'{split}_{label_name}_recall', metrics['recall'].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            average_metrics['acc'].append(metrics['acc'].item())
            average_metrics['auc'].append(metrics['auc'].item())
            average_metrics['f1'].append(metrics['f1'].item())
            average_metrics['precision'].append(metrics['precision'].item())
            average_metrics['recall'].append(metrics['recall'].item())

        self.log(f'{split}_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{split}_acc', np.mean(average_metrics['acc']), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{split}_auc', np.mean(average_metrics['auc']), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{split}_f1', np.mean(average_metrics['f1']), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{split}_precision', np.mean(average_metrics['precision']), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{split}_recall', np.mean(average_metrics['recall']), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def retrieval_pass(self, x):
        if len(x.shape) == 5:
            B, N, C, H, W = x.shape
            x = x.reshape(B*N, C, H, W)
            emb = self.model(x)
            emb = emb.reshape(B, N, -1)
            # emb = self.fc1(emb)
            emb = torch.mean(emb, dim=1)
        else:
            emb = self.model(x)
            # emb = self.fc1(emb)
        
        return emb
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
    