import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


class CustomModel(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.model = nn.Sequential(
            nn.Linear(config['emb_dim'], 4*config['emb_dim']),
            nn.ReLU(),
            nn.Linear(4*config['emb_dim'], config['emb_dim']),
            nn.ReLU(),
            nn.Linear(config['emb_dim'], config['num_classes'])
        )

        try:
            self.model = torch.compile(self.model, disable=not config['compile'])
        except:
            pass
    
    def compute_metrics(self, logits, y):
        metrics = {}
        logits = F.sigmoid(logits)
        y_pred = (logits > 0.5).float()
        y_true = y.float()
        tp = (y_pred * y_true).sum(dim=0)
        tn = ((1 - y_pred) * (1 - y_true)).sum(dim=0)
        fp = (y_pred * (1 - y_true)).sum(dim=0)
        fn = ((1 - y_pred) * y_true).sum(dim=0)
        metrics['tp'] = tp
        metrics['tn'] = tn
        metrics['fp'] = fp
        metrics['fn'] = fn
        metrics['precision'] = tp / (tp + fp + 1e-8)
        metrics['recall'] = tp / (tp + fn + 1e-8)
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'] + 1e-8)
        metrics['acc'] = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        return metrics
    
    def compute_loss(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits, y)
    
    def forward(self, x, y):
        logits = self.model(x)
        loss = self.compute_loss(logits, y)
        return logits, loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self(x, y)
        metrics = self.compute_metrics(logits, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_acc', metrics['acc'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_f1', metrics['f1'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_precision', metrics['precision'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_recall', metrics['recall'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self(x, y)
        metrics = self.compute_metrics(logits, y)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_acc', metrics['acc'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_f1', metrics['f1'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_precision', metrics['precision'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_recall', metrics['recall'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]