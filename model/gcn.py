import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAccuracy

class CustomModel(pl.LightningModule):
    def __init__(self, config, in_features=1024, hidden_features=1024, out_features=1024, num_classes=14, num_nodes=36):
        super().__init__()
        
        # image featurizer
        self.cnn = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.cnn.classifier = nn.Identity()
        # freeze weights of cnn
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # GCN model
        self.graph_model = nn.ModuleList([
            GCNConv(in_features, hidden_features),
            GCNConv(hidden_features, out_features),
        ])

        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(out_features)
        self.fc = nn.Linear(out_features, num_classes)

        # hyperparameters
        self.config = config
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_classes = num_classes
        self.num_nodes = num_nodes

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
        # extract features from the image
        with torch.no_grad():
            node_feats = self.cnn(batch.x)
        
        # run it through the graph model
        for i, layer in enumerate(self.graph_model):
            node_feats = layer(node_feats, batch.edge_index)
            if i < len(self.graph_model) - 1:
                node_feats = self.relu(node_feats)
        
        # average pooling
        node_feats = node_feats.reshape(-1, self.num_nodes, self.out_features)
        node_feats = self.layer_norm(node_feats)
        node_feats = node_feats.mean(dim=1)

        # for classification
        logits = self.fc(node_feats)
        loss = self.compute_loss(logits, batch.y)
        
        return logits, loss
    
    def training_step(self, batch, batch_idx):
        logits, loss = self(batch)

        # iterate over all lables
        for i, label_name in enumerate(self.lables):
            metrics = self.compute_metrics(logits[:, i], batch.y[:, i])
            self.log(f'train_{label_name}_acc', metrics['acc'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])
            self.log(f'train_{label_name}_f1', metrics['f1'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])
            self.log(f'train_{label_name}_precision', metrics['precision'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])
            self.log(f'train_{label_name}_recall', metrics['recall'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])
        metrics = self.compute_metrics(logits, batch.y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])
        self.log('train_acc', metrics['acc'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])
        self.log('train_f1', metrics['f1'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])
        self.log('train_precision', metrics['precision'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])
        self.log('train_recall', metrics['recall'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])

        return loss
    
    def validation_step(self, batch, batch_idx):
        logits, loss = self(batch)

        # iterate over all lables
        for i, label_name in enumerate(self.lables):
            metrics = self.compute_metrics(logits[:, i], batch.y[:, i])
            self.log(f'val_{label_name}_acc', metrics['acc'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])
            self.log(f'val_{label_name}_f1', metrics['f1'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])
            self.log(f'val_{label_name}_precision', metrics['precision'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])
            self.log(f'val_{label_name}_recall', metrics['recall'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])
        metrics = self.compute_metrics(logits, batch.y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])
        self.log('val_acc', metrics['acc'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])
        self.log('val_f1', metrics['f1'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])
        self.log('val_precision', metrics['precision'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])
        self.log('val_recall', metrics['recall'].mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=logits.shape[0])

        return loss

    @torch.no_grad()
    def retrieval_pass(self, batch):
        # extract features from the image
        node_feats = self.cnn(batch.x)
        
        # run it through the graph model
        for i, layer in enumerate(self.graph_model):
            node_feats = layer(node_feats, batch.edge_index)
            if i < len(self.graph_model) - 1:
                node_feats = self.relu(node_feats)
        
        # average pooling
        node_feats = node_feats.reshape(-1, self.num_nodes, self.out_features)
        node_feats = self.layer_norm(node_feats)
        node_feats = node_feats.mean(dim=1)

        return node_feats
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]
