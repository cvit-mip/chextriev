import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryAUROC
from model.matryoshka import Matryoshka_CE_Loss, MRL_Linear_Layer
from common_metrics import compute_metrics as retrieval_metrics


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight) #18x1024
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class AnaXnetGCN(nn.Module):
    def __init__(self, config, num_classes, anatomy_size=18, in_channel=1024):
        super(AnaXnetGCN, self).__init__()
        self.config = config
        anatomy_out = 1024
        self.num_classes = num_classes

        self.anatomy_gc1 = GraphConvolution(in_channel, 2048)
        self.anatomy_gc2 = GraphConvolution(2048, 1024)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=2)

        self.node_fc = nn.Sequential(
            nn.LayerNorm(anatomy_out),
            nn.AdaptiveAvgPool2d((anatomy_size, anatomy_out)),
            nn.Linear(anatomy_out, num_classes)
        )

        self.graph_fc = nn.Sequential(
            nn.LayerNorm(anatomy_out),
            nn.AdaptiveAvgPool1d(anatomy_out),
            nn.Linear(anatomy_out, num_classes)
        )

        #anatomy adjacency matrix
        anatomy_inp_name = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/anatomy_matrix.csv'
        anatomy_inp = torch.FloatTensor(pd.read_csv(anatomy_inp_name, sep='\t').values)
        adj_mat = torch.ones((anatomy_inp.shape[0]+1, anatomy_inp.shape[1]+1), dtype=torch.float32)
        adj_mat[:-1, :-1] = anatomy_inp
        self.anatomy_inp_tensor = nn.Parameter(adj_mat)
        self.global_feat = torch.nn.Parameter(torch.randn(1, 1, in_channel))
        nn.init.xavier_uniform_(self.global_feat)

    def anatomy_gcn(self, feature):
        anatomy_inp = feature
        N = anatomy_inp.shape[1]
        adj = self.anatomy_inp_tensor.detach()[:N, :N]
        x = self.anatomy_gc1(anatomy_inp, adj)
        x = self.relu(x)
        x = self.anatomy_gc2(x, adj)

        x = x.transpose(1, 2)
        x = torch.matmul(feature, x)
        x = self.softmax(x)
        x = torch.matmul(x, anatomy_inp)
        return x

    def forward(self, feature, global_feat=None):
        feature = torch.cat([feature, global_feat.unsqueeze(1)], dim=1) if global_feat is not None else feature
        anatomy = self.anatomy_gcn(feature)
        anatomy = anatomy.add(feature)
        
        node_emb = anatomy
        node_logits = self.node_fc(node_emb)

        graph_emb = torch.mean(anatomy, dim=1) if self.config['pool'] == 'mean' else node_emb[:, -1, :]
        graph_logits = self.graph_fc(graph_emb)
        
        return node_logits, graph_logits, graph_emb
    
    def retrieval_pass(self, feature, global_feat=None):
        feature = torch.cat([feature, global_feat.unsqueeze(1)], dim=1) if global_feat is not None else feature
        anatomy = self.anatomy_gcn(feature)
        anatomy = anatomy.add(feature)
        anatomy = torch.mean(anatomy, dim=1)
        return anatomy


class CustomModel(pl.LightningModule):
    def __init__(self, config, in_features=1024, out_features=1024, num_classes=9, num_nodes=18):
        super().__init__()
        print(f'GCN Conv Baseline using global features is {config["is_global_feat"]}' + ' in a constrastive manner' if config['contrastive'] else '')
        
        # image featurizer
        self.cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
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

        self.model = AnaXnetGCN(config=config, num_classes=num_classes, anatomy_size=num_nodes, in_channel=in_features)
        
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

    def forward(self, images, y_node, y_graph, global_feat=None, return_loss=True, return_emb=False):
        # images (batch_size, anatomy_size, 3, 224, 224)
        # y (batch_size, anatomy_size, num_classes)
        B, N, C, H, W = images.shape
        images = images.reshape(B*N, C, H, W)

        # run it through the image featurizer (B*N, in_features)
        node_feats = self.cnn(images)
        global_feat = self.cnn(global_feat) if global_feat is not None else None

        # reshape back to (batch_size, anatomy_size, in_features)
        node_feats = node_feats.reshape(B, N, -1)
        node_feats = node_feats if self.config['pool'] == 'mean' else torch.cat([node_feats, self.model.global_feat.expand(B, 1, -1)], dim=1)

        # for classification
        node_logits, graph_logits, graph_emb = self.model(node_feats, global_feat)
        if not return_loss:
            return (node_logits, graph_logits, graph_emb) if return_emb else (node_logits, graph_logits)
        
        loss_node = self.compute_loss(node_logits, y_node)
        loss_graph = self.compute_loss(graph_logits, y_graph)
        loss = loss_graph * self.graph_importance + loss_node * (1 - self.graph_importance)

        if self.config['contrastive']:
            positive_samples = []
            negative_samples = []
            for i, y_graph_i in enumerate(y_graph):
                common = (y_graph_i == y_graph).sum(dim=1)
                # pick the index of the least common label
                negative_samples.append(graph_emb[common.argmin()])
                # pick index of the most of common label (except itself)
                common[i] = -1
                positive_samples.append(graph_emb[common.argmax()])
            positive_samples = torch.stack(positive_samples)
            negative_samples = torch.stack(negative_samples)
            contrastive_loss = self.compute_contrastive_loss(self.hash_fn(graph_emb), self.hash_fn(positive_samples), self.hash_fn(negative_samples))
            loss = loss + self.graph_importance * 0.5 * contrastive_loss
        
        return (node_logits, graph_logits, graph_emb, loss) if return_emb else (node_logits, graph_logits, loss)
    
    def training_step(self, batch, batch_idx):
        y_node = batch['y']
        y_graph = (torch.sum(y_node, dim=1) > 0).float()
        global_feat = batch['global_feat'] if self.config['is_global_feat'] else None
        node_logits, graph_logits, loss = self(batch['node_feat'], y_node, y_graph, global_feat)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('graph_importance', self.graph_importance, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        y_node = batch['y']
        y_graph = (torch.sum(y_node, dim=1) > 0).float()
        global_feat = batch['global_feat'] if self.config['is_global_feat'] else None
        node_logits, graph_logits, graph_emb, loss = self(batch['node_feat'], y_node, y_graph, global_feat, return_emb=True)
        
        node_logits = node_logits[-1] if self.matryoshka else node_logits
        graph_logits = graph_logits[-1] if self.matryoshka else graph_logits

        self.val_epoch_end_outputs_node.append((node_logits.detach().cpu(), y_node.detach().cpu(), loss.detach().cpu()))
        self.val_epoch_end_outputs_graph.append((graph_logits.detach().cpu(), y_graph.detach().cpu(), graph_emb.detach().cpu()))

        return loss
    
    def test_step(self, batch, batch_idx):
        y_node = batch['y']
        y_graph = (torch.sum(y_node, dim=1) > 0).float()
        global_feat = batch['global_feat'] if self.config['is_global_feat'] else None
        node_logits, graph_logits, graph_emb, loss = self(batch['node_feat'], y_node, y_graph, global_feat, return_emb=True)

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
    def retrieval_pass(self, images, global_feat=None, is_concat=False):
        # images (batch_size, anatomy_size, 3, 224, 224)
        # y (batch_size, anatomy_size, num_classes)
        B, N, C, H, W = images.shape
        images = images.reshape(B*N, C, H, W)

        # run it through the image featurizer (B*N, in_features)
        node_feats = self.cnn(images)
        global_feat = self.cnn(global_feat) if global_feat is not None else None

        # reshape back to (batch_size, anatomy_size, in_features)
        node_feats = node_feats.reshape(B, N, -1)

        # for retrieval
        graph_emb = self.model.retrieval_pass(node_feats, global_feat)
        graph_emb = torch.cat((graph_emb, global_feat), dim=1) if is_concat and global_feat is not None else graph_emb
        graph_emb = self.hash_fn(graph_emb) if self.config['contrastive'] else graph_emb
        
        return graph_emb
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }
