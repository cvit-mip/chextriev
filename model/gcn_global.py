import numpy as np
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryAUROC

class CustomModel(pl.LightningModule):
    def __init__(self, config, in_features=1024, hidden_features=1024, out_features=1024, num_classes=14, num_nodes=18):
        super().__init__()
        self.lr = config['lr']

        model_paths = sorted(glob('/home/ssd_scratch/users/arihanth.srikar/checkpoints/mimic-cxr/densenet_all_images/*.ckpt'))
        model_criteria = 'auc'
        model_path = [md_name for md_name in model_paths if model_criteria in md_name][-1]
        weights = torch.load(model_path)['state_dict']
        weights = {k.replace('model.', ''): v for k, v in weights.items() if 'classifier' not in k}
        
        # image featurizer
        # self.cnn = densenet121(DenseNet121_Weights.DEFAULT)
        self.cnn = densenet121()
        self.cnn.classifier = nn.Identity()
        self.cnn.load_state_dict(weights)
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
        self.global_fc = nn.Sequential(
            nn.Linear(in_features+out_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
            )

        # hyperparameters
        self.config = config
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_classes = num_classes
        self.num_nodes = num_nodes

        self.calc_metrics = {
                'f1': BinaryF1Score(),
                'auc': BinaryAUROC(),
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
        
        self.val_epoch_end_outputs = []
        self.test_epoch_end_outputs = []

    def compute_metrics(self, logits, y):
        logits = logits.detach().cpu()
        y = y.detach().cpu()
        return {
            'f1': self.calc_metrics['f1'](logits, y),
            'auc': self.calc_metrics['auc'](logits, y),
            'acc': self.calc_metrics['acc'](logits, y),
            'recall': self.calc_metrics['recall'](logits, y),
            'precision': self.calc_metrics['precision'](logits, y),
        }

    def compute_loss(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits, y)

    def forward(self, sample, global_feature):
        # run it through the graph model
        node_feats = sample.x
        for i, layer in enumerate(self.graph_model):
            node_feats = layer(node_feats, sample.edge_index)
            if i < len(self.graph_model) - 1:
                node_feats = self.relu(node_feats)
        
        # average pooling
        node_feats = node_feats.reshape(-1, self.num_nodes, self.out_features)
        node_feats = self.layer_norm(node_feats)
        node_feats = node_feats.mean(dim=1)
        node_feats = torch.cat([node_feats, global_feature], dim=1)
        node_feats = self.global_fc(node_feats)

        # for classification
        logits = self.fc(node_feats)
        loss = self.compute_loss(logits, sample.y)
        
        return logits, loss
    
    def training_step(self, batch, batch_idx):
        sample, global_feature = batch
        logits, loss = self(sample, global_feature)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        sample, global_feature = batch
        logits, loss = self(sample, global_feature)

        self.val_epoch_end_outputs.append((logits.detach().cpu(), sample.y.detach().cpu(), loss.detach().cpu()))

        return loss
    
    def test_step(self, batch, batch_idx):
        sample, global_feature = batch
        logits, loss = self(sample, global_feature)

        self.test_epoch_end_outputs.append((logits.detach().cpu(), sample.y.detach().cpu(), loss.detach().cpu()))

        return loss
    
    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:

        logits, ys, losses = zip(*self.val_epoch_end_outputs)
        logits = torch.cat(logits)
        y = torch.cat(ys)
        loss = torch.mean(torch.tensor(list(losses)))

        self.common_end_epoch_function(logits, y, loss, 'val')
    
        self.val_epoch_end_outputs = []
    
    @torch.no_grad()
    def on_test_epoch_end(self):

        logits, ys, losses = zip(*self.test_epoch_end_outputs)
        logits = torch.cat(logits)
        y = torch.cat(ys)
        loss = torch.mean(torch.tensor(list(losses)))

        self.common_end_epoch_function(logits, y, loss, 'test')
    
        self.test_epoch_end_outputs = []
    
    
    @torch.no_grad()
    def common_end_epoch_function(self, logits, y, loss, split):

        average_metrics = {
            'acc': [],
            'auc': [],
            'f1': [],
            'precision': [],
            'recall': [],
        }
        # iterate over all lables
        for i, lable_name in enumerate(self.lables):
            metrics = self.compute_metrics(logits[:, i], y[:, i])

            self.log(f'{split}_{lable_name}_acc', metrics['acc'].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'{split}_{lable_name}_auc', metrics['auc'].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'{split}_{lable_name}_f1', metrics['f1'].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'{split}_{lable_name}_precision', metrics['precision'].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'{split}_{lable_name}_recall', metrics['recall'].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

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

    @torch.no_grad()
    def retrieval_pass(self, sample, global_feature):
        # run it through the graph model
        node_feats = sample.x
        for i, layer in enumerate(self.graph_model):
            node_feats = layer(node_feats, sample.edge_index)
            if i < len(self.graph_model) - 1:
                node_feats = self.relu(node_feats)
        
        # average pooling
        node_feats = node_feats.reshape(-1, self.num_nodes, self.out_features)
        node_feats = self.layer_norm(node_feats)
        node_feats = node_feats.mean(dim=1)
        node_feats = torch.cat([node_feats, global_feature], dim=1)
        node_feats = self.global_fc(node_feats)

        return node_feats, sample.y
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]
