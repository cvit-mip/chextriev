import numpy as np
import pandas as pd
from glob import glob
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


class MHA(nn.Module):
    '''
    Multi-head attention
    '''
    def __init__(self, hidden_dim: int=1024, n_head: int=16, dropout: float=0.0) -> None:
        super(MHA, self).__init__()
        self.n_embd = hidden_dim
        self.n_head = n_head
        self.dropout = dropout

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.kv = nn.Linear(hidden_dim, hidden_dim*2)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
    
    def forward(self, q, kv):
        B, T_q, C = q.size() # batch size, sequence length, embedding dimensionality (n_embd)
        B, T_kv, C = kv.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, (k, v) = self.q(q), self.kv(kv).split(self.n_embd, dim=2)
        q = q.view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T_kv, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T_kv, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # attention (B, nh, T_q, hs) x (B, nh, hs, T_kv) -> (B, nh, T_q, T_kv)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T_q, T_kv) x (B, nh, T_kv, hs) -> (B, nh, T_q, hs)
        y = y.transpose(1, 2).contiguous().view(B, T_q, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class AnaXnetGCN(nn.Module):
    def __init__(self, num_classes, anatomy_size=18, in_channel=1024, num_layers: int=2, dropout: float=0.0, matryoshka=False):
        super(AnaXnetGCN, self).__init__()
        anatomy_out = 1024
        self.num_classes = num_classes

        self.gcn_layers = nn.ModuleList([GraphConvolution(in_channel, in_channel) for _ in range(num_layers-1)]+[GraphConvolution(in_channel, anatomy_out)])

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

        self.node_attn_layers = nn.ModuleList([MHA(hidden_dim=in_channel, n_head=16, dropout=dropout) for _ in range(num_layers-1)]+[MHA(hidden_dim=anatomy_out, n_head=16, dropout=dropout)])
        self.graph_attn = MHA(hidden_dim=anatomy_out, n_head=16, dropout=dropout)

        self.fc_node = nn.Sequential(
            nn.LayerNorm(anatomy_out),
            nn.AdaptiveAvgPool2d((anatomy_size, anatomy_out)),
            nn.Linear(anatomy_out, num_classes) if not matryoshka else MRL_Linear_Layer(nesting_list=[8, 16, 32, 64, 128, 256, 512, 1024], 
                                                                                        num_classes=num_classes, efficient=True)
        )
        self.fc_graph = nn.Sequential(
            nn.LayerNorm(anatomy_out),
            nn.AdaptiveAvgPool1d(anatomy_out),
            nn.Linear(anatomy_out, num_classes) if not matryoshka else MRL_Linear_Layer(nesting_list=[8, 16, 32, 64, 128, 256, 512, 1024], 
                                                                                        num_classes=num_classes, efficient=True)
        )

        #anatomy adjacency matrix
        try:
            anatomy_inp_name = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/anatomy_matrix.csv'
            anatomy_inp = pd.read_csv(anatomy_inp_name, sep='\t')
        except:
            anatomy_inp_name = '/scratch/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/anatomy_matrix.csv'
            anatomy_inp = pd.read_csv(anatomy_inp_name, sep='\t')
        self.anatomy_inp_tensor = nn.Parameter(torch.FloatTensor(anatomy_inp.values))

    def anatomy_gcn(self, x, global_feat=None, adj=None):
        adj = self.anatomy_inp_tensor.detach() if adj is None else adj

        for i, (layer, node_attn) in enumerate(zip(self.gcn_layers, self.node_attn_layers)):
            # apply GCN
            x = self.dropout(layer(x, adj))
            
            # concatenate the global feature to the keys and values and pass it through the attention layer
            kv = torch.cat((x, global_feat.unsqueeze(1)), dim=1) if global_feat is not None else x
            x = node_attn(q=x, kv=kv)
            
            # apply non-linearity except for the last layer
            x = self.relu(x) if i != len(self.gcn_layers) - 1 else x

        return x

    def forward(self, feature, global_feat=None, return_emb=False):
        anatomy = self.anatomy_gcn(feature, global_feat)
        anatomy = anatomy.add(feature)
        
        # node level classification
        node_logits = self.fc_node(anatomy)

        # graph level classification
        kv = torch.cat((anatomy, global_feat.unsqueeze(1)), dim=1) if global_feat is not None else anatomy
        graph_emb = self.graph_attn(q=global_feat.unsqueeze(1), kv=kv).squeeze(1) if global_feat is not None else torch.mean(self.graph_attn(q=anatomy, kv=kv), dim=1)
        graph_logits = self.fc_graph(graph_emb)
        
        if return_emb:
            return node_logits, graph_logits, graph_emb

        return node_logits, graph_logits
    
    def retrieval_pass(self, feature, global_feat=None):
        anatomy = self.anatomy_gcn(feature, global_feat)
        anatomy = anatomy.add(feature)
        
        # graph level
        kv = torch.cat((anatomy, global_feat.unsqueeze(1)), dim=1) if global_feat is not None else anatomy
        graph_emb = self.graph_attn(q=global_feat.unsqueeze(1), kv=kv).squeeze(1) if global_feat is not None else torch.mean(kv, dim=1)
        
        return graph_emb


class CustomModel(pl.LightningModule):
    def __init__(self, config, in_features=1024, hidden_features=1024, out_features=1024, num_classes=9, num_nodes=18):
        super().__init__()
        print(f'Anaxnet multilayer attention using global features is {config["is_global_feat"]}')
        
        # image featurizer
        self.cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(2048, in_features)

        # freeze weights of cnn
        for param in self.cnn.parameters():
            param.requires_grad = False
        # unfreeze the fc layer
        for param in self.cnn.fc.parameters():
            param.requires_grad = True

        self.model = AnaXnetGCN(num_classes=num_classes, anatomy_size=num_nodes, in_channel=in_features, 
                                num_layers=config['num_layers'], dropout=config['dropout'], matryoshka=config['matryoshka'])

        # hyperparameters
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
        self.matryoshka = config['matryoshka']
        
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

    def forward(self, images, ys, has_global_feat=True, return_loss=True, return_emb=False):
        # images (batch_size, anatomy_size, 3, 224, 224)
        # y (batch_size, anatomy_size, num_classes)
        B, N, C, H, W = images.shape
        images = images.reshape(B*N, C, H, W)

        # run it through the image featurizer (B*N, in_features)
        node_feats = self.cnn(images)
        
        # reshape back to (batch_size, anatomy_size, in_features)
        node_feats = node_feats.reshape(B, N, -1)

        global_feat = node_feats[:, 0, :] if has_global_feat else None
        node_feats = node_feats[:, 1:, :]

        y_graph = ys[:, 0, :]
        y_node = ys[:, 1:, :]

        # for classification
        if return_emb:
            node_logits, graph_logits, graph_emb = self.model(node_feats, global_feat, return_emb)
        else:
            node_logits, graph_logits = self.model(node_feats, global_feat)
        if not return_loss:
            return (node_logits, graph_logits, graph_emb) if return_emb else (node_logits, graph_logits)
        
        loss_node = self.compute_loss(node_logits, y_node)
        loss_graph = self.compute_loss(graph_logits, y_graph)
        loss = loss_graph * self.graph_importance + loss_node * (1 - self.graph_importance)
        
        return (node_logits, graph_logits, graph_emb, loss) if return_emb else (node_logits, graph_logits, loss)
    
    def training_step(self, batch, batch_idx):
        node_logits, graph_logits, loss = self(batch['images'], batch['y_9'], has_global_feat=self.config['is_global_feat'])

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('graph_importance', self.graph_importance, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        node_logits, graph_logits, graph_emb, loss = self(batch['images'], batch['y_9'], has_global_feat=self.config['is_global_feat'], return_emb=True)
        
        node_logits = node_logits[-1] if self.matryoshka else node_logits
        graph_logits = graph_logits[-1] if self.matryoshka else graph_logits

        self.val_epoch_end_outputs_node.append((node_logits.detach().cpu(), batch['y_9'][:, 1:, :].detach().cpu(), loss.detach().cpu()))
        self.val_epoch_end_outputs_graph.append((graph_logits.detach().cpu(), batch['y_9'][:, 0, :].detach().cpu(), graph_emb.detach().cpu()))

        return loss
    
    def test_step(self, batch, batch_idx):
        node_logits, graph_logits, graph_emb, loss = self(batch['images'], batch['y_9'], has_global_feat=self.config['is_global_feat'], return_emb=True)

        node_logits = node_logits[-1] if self.matryoshka else node_logits
        graph_logits = graph_logits[-1] if self.matryoshka else graph_logits

        self.test_epoch_end_outputs_node.append((node_logits.detach().cpu(), batch['y_9'][:, 1:, :].detach().cpu(), loss.detach().cpu()))
        self.test_epoch_end_outputs_graph.append((graph_logits.detach().cpu(), batch['y_9'][:, 0, :].detach().cpu(), graph_emb.detach().cpu()))

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
    def retrieval_pass(self, images, has_global_feat=True, is_concat=False):
        # images (batch_size, anatomy_size, 3, 224, 224)
        # y (batch_size, anatomy_size, num_classes)
        B, N, C, H, W = images.shape
        images = images.reshape(B*N, C, H, W)

        # run it through the image featurizer (B*N, in_features)
        node_feats = self.cnn(images)
        
        # reshape back to (batch_size, anatomy_size, in_features)
        node_feats = node_feats.reshape(B, N, -1)

        global_feat = node_feats[:, 0, :] if has_global_feat else None
        node_feats = node_feats[:, 1:, :]

        # for retrieval
        graph_emb = self.model.retrieval_pass(node_feats, global_feat)
        graph_emb = torch.cat((graph_emb, global_feat), dim=1) if is_concat and global_feat is not None else graph_emb
        
        return graph_emb
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config['lr'], weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }
