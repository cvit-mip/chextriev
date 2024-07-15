import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryAUROC

from model.anaxnet_attn_multilayer import CustomModel as AnaxnetMultilayerAttn
from common_metrics import compute_metrics as retrieval_metrics


class CustomModel(pl.LightningModule):
    def __init__(self, config, in_features=1024, hidden_features=1024, out_features=1024, num_classes=9, num_nodes=18):
        super().__init__()
        print(f'Training constrastive hash function using Anaxnet multilayer attention whose global features is {config["is_global_feat"]}')

        self.model = AnaxnetMultilayerAttn(config, in_features, hidden_features, out_features, num_classes, num_nodes)
        self.hash_fn = nn.Sequential(
            nn.Linear(out_features, out_features*4),
            nn.ReLU(),
            nn.Linear(out_features*4, out_features),
            nn.ReLU(),
            nn.Linear(out_features, config['hash_bits']),
        )
        self.criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))

        # hyperparameters for anaxnet
        self.config = config
        self.lr = config['lr']
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.graph_importance = config['graph_importance']
        self.hash_bits = config['hash_bits']
        
        self.save_hyperparameters()

        self.retrieval_metrics = retrieval_metrics
        self.calc_metrics = {
                'f1': BinaryF1Score(),
                'auc': BinaryAUROC(),
                'acc': BinaryAccuracy(),
                'recall': BinaryRecall(),
                'precision': BinaryPrecision(),
            }
        
        self.node_lables = [
            'lung opacity', 
            'pleural effusion', 
            'atelectasis', 
            'enlarged cardiac silhouette',
            'pulmonary edema/hazy opacity', 
            'pneumothorax', 
            'consolidation', 
            'fluid overload/heart failure', 
            'pneumonia']
        self.graph_lables = [ 
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

        self.val_epoch_end_outputs_node = []
        self.val_epoch_end_outputs_graph = []
        self.test_epoch_end_outputs_node = []
        self.test_epoch_end_outputs_graph = []

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

    def compute_loss(self, anchor_emb, postive_emb, negative_emb):
        return self.criterion(anchor_emb, postive_emb, negative_emb)

    def common_step(self, sample_data, return_loss=False):
        y_node = sample_data['y']
        y_graph = (torch.sum(y_node, dim=1) > 0).float()
        global_feat = sample_data['global_feat'] if self.config['is_global_feat'] else None
        return self.model(sample_data['node_feat'], y_node, y_graph, global_feat, return_loss, return_emb=True)
    
    def forward(self, anchor, positive, negative):
        anchor_emb = self.hash_fn(anchor)
        positive_emb = self.hash_fn(positive)
        negative_emb = self.hash_fn(negative)
        return self.compute_loss(anchor_emb, positive_emb, negative_emb)
    
    def training_step(self, batch, batch_idx):
        _, _, anchor, anchor_loss = self.common_step(batch['anchor'], return_loss=True)
        _, _, positive, positive_loss = self.common_step(batch['positive'], return_loss=True)
        _, _, negative, negative_loss = self.common_step(batch['negative'], return_loss=True)
        
        contrastive_loss = self(anchor, positive, negative)
        # loss = anchor_loss + positive_loss + negative_loss + contrastive_loss
        loss = contrastive_loss

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_classification_loss', anchor_loss + positive_loss + negative_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_contrastive_loss', contrastive_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('graph_importance', self.graph_importance, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        anchor_node, anchor_graph, anchor, anchor_loss = self.common_step(batch['anchor'], return_loss=True)
        _, _, positive = self.common_step(batch['positive'], return_loss=False)
        _, _, negative = self.common_step(batch['negative'], return_loss=False)
        
        contrastive_loss = self(anchor, positive, negative)
        loss = anchor_loss + contrastive_loss

        self.val_epoch_end_outputs_node.append((anchor_node.detach().cpu(), batch['anchor']['y'].detach().cpu(), anchor_loss.detach().cpu()))
        self.val_epoch_end_outputs_graph.append((anchor_graph.detach().cpu(), (torch.sum(batch['anchor']['y'], dim=1) > 0).float().detach().cpu(), anchor.detach().cpu()))

        return loss
    
    def test_step(self, batch, batch_idx):
        anchor_node, anchor_graph, anchor, anchor_loss = self.common_step(batch['anchor'], return_loss=True)
        _, _, positive = self.common_step(batch['positive'], return_loss=False)
        _, _, negative = self.common_step(batch['negative'], return_loss=False)
        
        contrastive_loss = self(anchor, positive, negative)
        loss = anchor_loss + contrastive_loss

        self.test_epoch_end_outputs_node.append((anchor_node.detach().cpu(), batch['anchor']['y'].detach().cpu(), anchor_loss.detach().cpu()))
        self.test_epoch_end_outputs_graph.append((anchor_graph.detach().cpu(), (torch.sum(batch['anchor']['y'], dim=1) > 0).float().detach().cpu(), anchor.detach().cpu()))

        return loss
    
    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
    
        node_logits, node_ys, node_losses = zip(*self.val_epoch_end_outputs_node)
        node_logits = torch.cat(node_logits)
        node_y = torch.cat(node_ys)
        loss = torch.mean(torch.tensor(list(node_losses)))
        
        graph_logits, graph_ys, graph_emb = zip(*self.val_epoch_end_outputs_graph)
        graph_logits = torch.cat(graph_logits)
        graph_y = torch.cat(graph_ys)
        graph_emb = torch.cat(graph_emb)

        self.common_end_epoch_function(node_logits, node_y, loss, graph_logits, graph_y, graph_emb, 'val')
    
        self.val_epoch_end_outputs_node = []
        self.val_epoch_end_outputs_graph = []
    
    @torch.no_grad()
    def on_test_epoch_end(self):

        node_logits, node_ys, node_losses = zip(*self.test_epoch_end_outputs_node)
        node_logits = torch.cat(node_logits)
        node_y = torch.cat(node_ys)
        loss = torch.mean(torch.tensor(list(node_losses)))
        
        graph_logits, graph_ys, graph_emb = zip(*self.test_epoch_end_outputs_graph)
        graph_logits = torch.cat(graph_logits)
        graph_y = torch.cat(graph_ys)
        graph_emb = torch.cat(graph_emb)

        self.common_end_epoch_function(node_logits, node_y, loss, graph_logits, graph_y, graph_emb, 'test')
    
        self.test_epoch_end_outputs_node = []
        self.test_epoch_end_outputs_graph = []
    
    
    @torch.no_grad()
    def common_end_epoch_function(self, node_logits, node_y, loss, graph_logits, graph_y, graph_emb, split):

        average_metrics = {
            'acc': [],
            'auc': [],
            'f1': [],
            'precision': [],
            'recall': [],
        }

        _, mAP, mHR, mRR = self.retrieval_metrics(self.config, split, graph_emb.numpy(), graph_y.numpy(), 
                                                  self.node_lables, top_k=10, dist_metric='cosine', is_save=False)
        self.log(f'{split}_mAP', mAP, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{split}_mHR', mHR, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'{split}_mRR', mRR, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # graph level classification
        for i, lable_name in enumerate(self.node_lables):
            metrics = self.compute_metrics(graph_logits[:, i], graph_y[:, i])

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
        
        average_metrics = {
            'acc': [],
            'auc': [],
            'f1': [],
            'precision': [],
            'recall': [],
        }
        
        # node level classification
        for i, lable_name in enumerate(self.node_lables):
            metrics = self.compute_metrics(node_logits[:, :, i], node_y[:, :, i])

            self.log(f'node_{split}_{lable_name}_acc', metrics['acc'].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'node_{split}_{lable_name}_auc', metrics['auc'].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'node_{split}_{lable_name}_f1', metrics['f1'].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'node_{split}_{lable_name}_precision', metrics['precision'].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f'node_{split}_{lable_name}_recall', metrics['recall'].item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            average_metrics['acc'].append(metrics['acc'].item())
            average_metrics['auc'].append(metrics['auc'].item())
            average_metrics['f1'].append(metrics['f1'].item())
            average_metrics['precision'].append(metrics['precision'].item())
            average_metrics['recall'].append(metrics['recall'].item())

        self.log(f'node_{split}_acc', np.mean(average_metrics['acc']), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'node_{split}_auc', np.mean(average_metrics['auc']), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'node_{split}_f1', np.mean(average_metrics['f1']), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'node_{split}_precision', np.mean(average_metrics['precision']), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f'node_{split}_recall', np.mean(average_metrics['recall']), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    @torch.no_grad()
    def retrieval_pass(self, batch, global_feat=None, is_concat=False):
        y_node = batch['anchor']['y']
        y_graph = (torch.sum(y_node, dim=1) > 0).float()
        global_feat = batch['anchor']['global_feat'] if self.config['is_global_feat'] else None
        _, graph_emb = self.model(batch['anchor']['node_feat'], y_node, y_graph, global_feat, return_loss=False)

        graph_emb = self.hash_fn(graph_emb)
        graph_emb = torch.cat((graph_emb, global_feat), dim=1) if is_concat and global_feat is not None else graph_emb
        
        return graph_emb
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }
