import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, densenet121, DenseNet121_Weights, efficientnet_b0, EfficientNet_B0_Weights
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryAUROC
from model.matryoshka import Matryoshka_CE_Loss, MRL_Linear_Layer
from common_metrics import compute_metrics as retrieval_metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingLR
from Res2Net.res2net import res2net50


class CustomModel(pl.LightningModule):
    def __init__(self, config, in_features=1024, out_features=1024, num_classes=9, num_nodes=18):
        super().__init__()
        print(f'Graph Tansformer using global features is {config["is_global_feat"]}' + ' in a constrastive manner' if config['contrastive'] else '')
        
        # image featurizer
        if 'resnet' in config['image_featuriser']:
            self.cnn = resnet50(weights=ResNet50_Weights.DEFAULT) if not config["multiscale"] else res2net50(pretrained=True)
            self.cnn.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, in_features),
            )
            # freeze weights of cnn
            for param in self.cnn.parameters():
                param.requires_grad = False
            for param in self.cnn.fc.parameters():
                param.requires_grad = True
        
        elif 'densenet' in config['image_featuriser']:
            self.cnn = densenet121(weights=DenseNet121_Weights.DEFAULT)
            self.cnn.classifier = nn.Identity()
            # freeze weights of cnn
            # for param in self.cnn.parameters():
            #     param.requires_grad = False

        elif 'efficientnet' in config['image_featuriser']:
            self.cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.cnn.classifier = nn.Sequential(
                nn.Linear(1280, 1024),
                nn.ReLU(),
                nn.Linear(1024, in_features),
            )
            # freeze weights of cnn
            # for param in self.cnn.parameters():
            #     param.requires_grad = False
            # for param in self.cnn.classifier.parameters():
            #     param.requires_grad = True

        else:
            raise NotImplementedError(f'Image featuriser {config["image_featuriser"]} not implemented')

        encoder_layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=8, dropout=config['dropout'], batch_first=True)
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'], norm=nn.LayerNorm(in_features))

        self.fc = nn.Sequential(
            nn.LayerNorm(out_features),
            nn.Linear(out_features, num_classes) if not config['matryoshka'] else MRL_Linear_Layer(nesting_list=[8, 16, 32, 64, 128, 256, 512, 1024], 
                                                                                                   num_classes=num_classes, efficient=True)
        )

        if config['cls']:
            print('Using CLS token')
            self.register_buffer('cls_token', nn.Parameter(torch.zeros(1, out_features), requires_grad=True))
            nn.init.xavier_uniform_(self.cls_token)
        
        #anatomy adjacency matrix
        adj_mat_file = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/anatomy_matrix.csv'
        adj_mat = torch.from_numpy(pd.read_csv(adj_mat_file, sep='\t').to_numpy()).int()
        N, M = adj_mat.shape
        if config['is_global_feat']:
            assert N == M, 'Adjacency matrix should be square'
            N += 1 if not config['cls'] else 2
            M += 1 if not config['cls'] else 2
            temp_adj_mat = torch.ones((N, M))
            if not config['fully_connected']:
                temp_adj_mat[:-1, :-1] = adj_mat
        else:
            temp_adj_mat = adj_mat if not config['fully_connected'] else torch.ones_like(adj_mat)
        self.register_buffer('adj_mat', temp_adj_mat)
        
        if config['contrastive']:
            hash_bits = config['hash_bits']
            self.hash_fn = nn.Sequential(
                nn.Linear(out_features, max(out_features//2, hash_bits)),
                nn.ReLU(),
                nn.Linear(max(out_features//2, hash_bits), hash_bits),
            )

        # hyperparameters
        self.config = config
        self.lr = config['lr']
        self.in_features = in_features
        self.out_features = out_features
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.graph_importance = config['graph_importance']
        self.matryoshka = config['matryoshka']
        self.contrastive = config['contrastive']
        
        self.save_hyperparameters()

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

        self.criterion = Matryoshka_CE_Loss(is_functional=True) if config['matryoshka'] else F.binary_cross_entropy_with_logits
        self.contrastive_criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
        self.retrieval_metrics = retrieval_metrics

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
        return self.criterion(logits, y)
    
    def compute_contrastive_loss(self, anchor, positive, negative):
        return self.contrastive_criterion(anchor, positive, negative)

    def forward(self, images, y_node, y_graph, global_feat=None, return_loss=True, return_emb=False, masks=None, retrieval=False):
        # images (batch_size, anatomy_size, 3, 224, 224)
        # y (batch_size, anatomy_size, num_classes)
        images = torch.cat([images, global_feat.unsqueeze(1)], dim=1) if global_feat is not None else images
        B, N, C, H, W = images.shape
        images = images.reshape(B*N, C, H, W)

        # run it through the image featurizer (B*N, in_features)
        node_feats = self.cnn(images)

        # reshape back to (batch_size, anatomy_size, in_features)
        node_feats = node_feats.reshape(B, N, -1)

        # run model
        node_feats = torch.cat([node_feats, self.cls_token.expand(B, -1, -1)], dim=1) if self.config['cls'] else node_feats
        # node_feats = self.model(node_feats, mask=self.adj_mat.bool())
        node_feats = self.model(node_feats)
        
        # for classification
        logits = self.fc(node_feats)
        if global_feat is not None or self.config['cls']:
            node_logits = logits[:, :self.num_nodes, :]
            graph_emb = node_feats[:, -1, :] if self.config['pool'] == 'attn' else torch.mean(node_feats[:, :self.num_nodes, :], dim=1)
        else:
            node_logits = logits
            graph_emb = torch.mean(node_feats, dim=1)
        graph_logits = self.fc(graph_emb)
        
        if not return_loss:
            return (node_logits, graph_logits, graph_emb) if return_emb else (node_logits, graph_logits)
        
        loss_node = self.compute_loss(node_logits, y_node)
        loss_graph = self.compute_loss(graph_logits, y_graph)
        loss = loss_graph * self.graph_importance + loss_node * (1 - self.graph_importance)
        
        return (node_logits, graph_logits, graph_emb, loss) if return_emb else (node_logits, graph_logits, loss)
    
    def training_step(self, batch, batch_idx):
        y_node = batch['y']
        y_graph = (torch.sum(y_node, dim=1) > 0).float()
        global_feat = batch['global_feat'] if self.config['is_global_feat'] else None
        node_logits, graph_logits, loss = self(batch['node_feat'], y_node, y_graph, global_feat, masks=batch['masked_img'])

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('graph_importance', self.graph_importance, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        y_node = batch['y']
        y_graph = (torch.sum(y_node, dim=1) > 0).float()
        global_feat = batch['global_feat'] if self.config['is_global_feat'] else None
        node_logits, graph_logits, graph_emb, loss = self(batch['node_feat'], y_node, y_graph, global_feat, return_emb=True, masks=batch['masked_img'])
        
        node_logits = node_logits[-1] if self.matryoshka else node_logits
        graph_logits = graph_logits[-1] if self.matryoshka else graph_logits

        self.val_epoch_end_outputs_node.append((node_logits.detach().cpu(), y_node.detach().cpu(), loss.detach().cpu()))
        self.val_epoch_end_outputs_graph.append((graph_logits.detach().cpu(), y_graph.detach().cpu(), graph_emb.detach().cpu()))

        return loss
    
    def test_step(self, batch, batch_idx):
        y_node = batch['y']
        y_graph = (torch.sum(y_node, dim=1) > 0).float()
        global_feat = batch['global_feat'] if self.config['is_global_feat'] else None
        node_logits, graph_logits, graph_emb, loss = self(batch['node_feat'], y_node, y_graph, global_feat, return_emb=True, masks=batch['masked_img'])

        node_logits = node_logits[-1] if self.matryoshka else node_logits
        graph_logits = graph_logits[-1] if self.matryoshka else graph_logits

        self.test_epoch_end_outputs_node.append((node_logits.detach().cpu(), y_node.detach().cpu(), loss.detach().cpu()))
        self.test_epoch_end_outputs_graph.append((graph_logits.detach().cpu(), y_graph.detach().cpu(), graph_emb.detach().cpu()))

        return loss
    
    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
    
        node_logits, node_y, node_losses = zip(*self.val_epoch_end_outputs_node)
        node_logits = torch.cat(node_logits)
        node_y = torch.cat(node_y)
        loss = torch.mean(torch.tensor(list(node_losses)))
        
        graph_logits, graph_y, graph_emb = zip(*self.val_epoch_end_outputs_graph)
        graph_logits = torch.cat(graph_logits)
        graph_y = torch.cat(graph_y)
        graph_emb = torch.cat(graph_emb)

        self.common_end_epoch_function(node_logits, node_y, loss, graph_logits, graph_y, graph_emb, 'val')
    
        self.val_epoch_end_outputs_node = []
        self.val_epoch_end_outputs_graph = []
    
    @torch.no_grad()
    def on_test_epoch_end(self):

        node_logits, node_y, node_losses = zip(*self.test_epoch_end_outputs_node)
        node_logits = torch.cat(node_logits)
        node_y = torch.cat(node_y)
        loss = torch.mean(torch.tensor(list(node_losses)))
        
        graph_logits, graph_y, graph_emb = zip(*self.test_epoch_end_outputs_graph)
        graph_logits = torch.cat(graph_logits)
        graph_y = torch.cat(graph_y)
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

        # retrieval
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
    def retrieval_pass(self, images, global_feat=None):
        _, _, graph_emb = self(images, None, None, global_feat=global_feat if self.config['is_global_feat'] else None, 
                               return_loss=False, return_emb=True, masks=None, retrieval=False)
        
        global_emb = self.cnn(global_feat) if global_feat is not None and self.config['concat_global_feature'] else None
        graph_emb = torch.cat([graph_emb, global_emb], dim=1) if global_emb is not None else graph_emb
        
        return graph_emb
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config['lr'])
        if self.config['lr_scheduler'] == 'one_cycle':
            scheduler = OneCycleLR(optimizer, max_lr=self.config['lr'], total_steps=self.config['num_steps'])
            scheduler = {'scheduler': scheduler, 'interval': 'step'}
        elif self.config['lr_scheduler'] == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_steps']//2, eta_min=self.config['lr']/100)
            scheduler = {'scheduler': scheduler, 'interval': 'step'}
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
            scheduler = {'scheduler': scheduler, 'monitor': 'val_loss'}
        
        return [optimizer], scheduler
    