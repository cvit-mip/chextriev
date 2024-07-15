import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_transformer.graph_transformer_pytorch import GraphTransformer

import pytorch_lightning as pl
from torchvision.models import densenet121, DenseNet121_Weights

from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAccuracy

os.environ['TORCH_HOME'] = '/ssd_scratch/users/arihanth.srikar'


class CustomModel(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.lr = config['lr']

        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.model.classifier = nn.Identity()
        # model is not trainable
        for param in self.model.parameters():
            param.requires_grad = False
        
        # if embedding dimension is not 1024, compress it
        self.compress = nn.Linear(1024, config['emb_dim']) if config['emb_dim'] != 1024 else nn.Identity()
        self.fc = nn.Linear(config['emb_dim'], config['num_classes'])

        self.graph_model = GraphTransformer(
            dim = config['emb_dim'],
            depth = config['num_layers'],
            edge_dim=config['edge_dim'],
            with_feedforwards=True,
            gated_residual=True,
            abs_pos_emb=True,
            accept_adjacency_matrix=True,
        )

        self.model = torch.compile(self.model, disable=not config['compile'])
        self.graph_model = torch.compile(self.graph_model, disable=not config['compile'])

        self.save_hyperparameters()
    
        self.calc_metrics = {
                'f1': BinaryF1Score(),
                'acc': BinaryAccuracy(),
                'recall': BinaryRecall(),
                'precision': BinaryPrecision(),
            }
        self.lables = [ 'Atelectasis',
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
    
    def compute_metrics(self, logits, y):
        logits = logits.detach().cpu()
        y = y.detach().cpu()
        return {
            'f1': self.calc_metrics['f1'](logits, y),
            'acc': self.calc_metrics['acc'](logits, y),
            'recall': self.calc_metrics['recall'](logits, y),
            'precision': self.calc_metrics['precision'](logits, y),
        }
    
    def compute_loss(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits, y)
    
    def forward(self, batch):
        img, node_idx, node_features, edge_feats, adj_mat, mask, labels = batch
        
        # entire image embedding
        global_emb = self.model(img)
        global_emb = self.compress(global_emb)

        # obtain node features
        b, n, c, h, w = node_features.shape
        node_features = node_features.view(b*n, c, h, w)
        node_features = self.model(node_features)
        node_features = node_features.view(b, n, -1)
        node_features = self.compress(node_features)
        
        # graph embedding 
        nodes, edges = self.graph_model(node_features, edge_feats, adj_mat=adj_mat, mask=mask, positions=node_idx)
        
        # global mean pooling
        global_emb = (global_emb + torch.sum(nodes, dim=1)) / (n + 1)
        logits = self.fc(global_emb)

        loss = self.compute_loss(logits, labels)
        return logits, loss
    
    def training_step(self, batch, batch_idx):
        _, _, _, _, _, _, y = batch
        logits, loss = self(batch)

        # iterate over all lables
        for i, label_name in enumerate(self.lables):
            metrics = self.compute_metrics(logits[:, i], y[:, i])
            self.log(f'train_{label_name}_acc', metrics['acc'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'train_{label_name}_f1', metrics['f1'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'train_{label_name}_precision', metrics['precision'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'train_{label_name}_recall', metrics['recall'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        metrics = self.compute_metrics(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_acc', metrics['acc'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_f1', metrics['f1'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_precision', metrics['precision'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_recall', metrics['recall'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        _, _, _, _, _, _, y = batch
        logits, loss = self(batch)
        
        # iterate over all lables
        for i, label_name in enumerate(self.lables):
            metrics = self.compute_metrics(logits[:, i], y[:, i])
            self.log(f'val_{label_name}_acc', metrics['acc'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'val_{label_name}_f1', metrics['f1'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'val_{label_name}_precision', metrics['precision'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'val_{label_name}_recall', metrics['recall'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        metrics = self.compute_metrics(logits, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_acc', metrics['acc'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_f1', metrics['f1'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_precision', metrics['precision'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_recall', metrics['recall'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    @torch.no_grad()
    def retrieval_pass(self, batch):
        img, node_idx, node_features, edge_feats, adj_mat, mask, labels = batch
        
        # entire image embedding
        global_emb = self.model(img)
        global_emb = self.compress(global_emb)

        # obtain node features
        b, n, c, h, w = node_features.shape
        node_features = node_features.view(b*n, c, h, w)
        node_features = self.model(node_features)
        node_features = node_features.view(b, n, -1)
        node_features = self.compress(node_features)
        
        # graph embedding 
        nodes, edges = self.graph_model(node_features, edge_feats, adj_mat=adj_mat, mask=mask, positions=node_idx)
        
        # global mean pooling
        global_emb = (global_emb + torch.sum(nodes, dim=1)) / (n + 1)
        logits = self.fc(global_emb)

        return logits, labels
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]