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

    def __init__(self, in_features, out_features, bias=False, dropout=0.0):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout)
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
            return self.dropout(output + self.bias)
        else:
            return self.dropout(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class MHA(nn.Module):
    '''
    Multi-head attention
    '''
    def __init__(self, hidden_dim: int=1024, n_head: int=16, dropout: float=0.0, attn_type: str='graph', adj_mat=None) -> None:
        super(MHA, self).__init__()
        self.n_embd = hidden_dim
        self.n_head = n_head
        self.dropout = dropout

        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.kv = nn.Linear(hidden_dim, hidden_dim*2, bias=False)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        if attn_type == 'node':
            N, M = adj_mat.shape
            assert N == M, 'Adjacency matrix must be square'
            mask = torch.ones(N, N+1)
            mask[:, :-1] = adj_mat
            self.register_buffer('mask', mask.view(1, 1, N, N+1))
        else:
            self.register_buffer('mask', None)
    
    def forward(self, q, kv):
        B, T_q, C = q.size() # batch size, sequence length, embedding dimensionality (n_embd)
        B, T_kv, C = kv.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q, kv = self.layer_norm(q), self.layer_norm(kv)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, (k, v) = self.q(q), self.kv(kv).split(self.n_embd, dim=2)
        q = q.view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T_kv, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T_kv, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # attention (B, nh, T_q, hs) x (B, nh, hs, T_kv) -> (B, nh, T_q, T_kv)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.mask is not None:
            att = att.masked_fill(self.mask[:, :, :T_q, :T_kv] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T_q, T_kv) x (B, nh, T_kv, hs) -> (B, nh, T_q, hs)
        y = y.transpose(1, 2).contiguous().view(B, T_q, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class MLP(nn.Module):

    def __init__(self, d, dropout=0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d)
        self.c_fc    = nn.Linear(d, 4 * d, bias=False)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * d, d, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class AnaXnetGCN(nn.Module):
    def __init__(self, num_classes, anatomy_size=18, in_channel=1024, anatomy_out=1024, num_layers: int=2, dropout: float=0.0, matryoshka: bool=False, pool: str='attn', minimalistic: bool=False):
        super(AnaXnetGCN, self).__init__()
        self.num_classes = num_classes
        self.pool = pool
        self.minimalistic = minimalistic

        #anatomy adjacency matrix
        adj_mat_file = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/anatomy_matrix.csv'
        adj_mat = torch.FloatTensor(pd.read_csv(adj_mat_file, sep='\t').values)

        self.gcn_layers = nn.ModuleList([GraphConvolution(in_channel, in_channel, dropout=dropout) for _ in range(num_layers-1)]+[GraphConvolution(in_channel, anatomy_out, dropout=dropout)])

        self.node_attn_layers = nn.ModuleList([MHA(hidden_dim=in_channel, n_head=16, dropout=dropout, attn_type='node', adj_mat=adj_mat) for _ in range(num_layers-1)]+[MHA(hidden_dim=anatomy_out, n_head=16, dropout=dropout, attn_type='node', adj_mat=adj_mat)])
        self.graph_attn = MHA(hidden_dim=anatomy_out, n_head=16, dropout=dropout, attn_type='graph', adj_mat=adj_mat)

        self.mlp = nn.ModuleList([MLP(in_channel, dropout=dropout)]+[MLP(anatomy_out, dropout=dropout) for _ in range(num_layers-1)])

        self.fc_node = nn.Sequential(
            nn.LayerNorm(anatomy_out),
            nn.Linear(anatomy_out, num_classes) if not matryoshka else MRL_Linear_Layer(nesting_list=[8, 16, 32, 64, 128, 256, 512, 1024], 
                                                                                        num_classes=num_classes, efficient=True)
        )
        self.fc_graph = nn.Sequential(
            nn.LayerNorm(anatomy_out),
            nn.Linear(anatomy_out, num_classes) if not matryoshka else MRL_Linear_Layer(nesting_list=[8, 16, 32, 64, 128, 256, 512, 1024], 
                                                                                        num_classes=num_classes, efficient=True)
        )

        self.anatomy_inp_tensor = nn.Parameter(adj_mat)

    def block(self, x, global_feat=None, adj=None):
        adj = self.anatomy_inp_tensor.detach() if adj is None else adj

        x_resid = x.clone()
        for _, (gcn_conv, node_attn, mlp) in enumerate(zip(self.gcn_layers, self.node_attn_layers, self.mlp)):
            # GCN Convolution
            x = gcn_conv(x, adj)
            
            # concatenate the global feature to the keys and values and pass it through the attention layer
            kv = torch.cat((x, global_feat.unsqueeze(1)), dim=1) if global_feat is not None else x
            x = node_attn(q=x, kv=kv)

            # apply MLP
            x = mlp(x)
            
            # residual connection
            x = x + x_resid

        return x

    def forward(self, feature, global_feat=None, return_emb=False):
        anatomy = self.block(feature, global_feat) if not self.minimalistic else feature
        
        # node level classification
        node_logits = self.fc_node(anatomy)

        # graph level classification
        kv = torch.cat((anatomy, global_feat.unsqueeze(1)), dim=1) if global_feat is not None else anatomy
        graph_emb = self.graph_attn(q=global_feat.unsqueeze(1), kv=kv).squeeze(1) if global_feat is not None and self.pool == 'attn' else torch.mean(kv, dim=1)
        graph_logits = self.fc_graph(graph_emb)
        
        if return_emb:
            return node_logits, graph_logits, graph_emb

        return node_logits, graph_logits
    
    def retrieval_pass(self, feature, global_feat=None):
        anatomy = self.block(feature, global_feat) if not self.minimalistic else feature
        
        # graph level
        kv = torch.cat((anatomy, global_feat.unsqueeze(1)), dim=1) if global_feat is not None else anatomy
        graph_emb = self.graph_attn(q=global_feat.unsqueeze(1), kv=kv).squeeze(1) if global_feat is not None and self.pool == 'attn' else torch.mean(kv, dim=1)
        
        return graph_emb


class CustomModel(pl.LightningModule):
    def __init__(self, config, in_features=1024, out_features=1024, num_classes=9, num_nodes=18):
        super().__init__()
        print(f'xFactor using global features is {config["is_global_feat"]}' + ' in a constrastive manner' if config['contrastive'] else '')
        
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

        self.model = AnaXnetGCN(num_classes=num_classes, anatomy_size=num_nodes, in_channel=in_features, anatomy_out=out_features,
                                num_layers=config['num_layers'], dropout=config['dropout'], matryoshka=config['matryoshka'], 
                                pool=config['pool'], minimalistic=config['minimalistic'])
        
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

    def forward(self, images, y_node, y_graph, global_feat=None, return_loss=True, return_emb=False, masks=None):
        # images (batch_size, anatomy_size, 3, 224, 224)
        # y (batch_size, anatomy_size, num_classes)
        B, N, C, H, W = images.shape
        images = images.reshape(B*N, C, H, W)

        # run it through the image featurizer (B*N, in_features)
        node_feats = self.cnn(images)
        global_images = global_feat.clone() if global_feat is not None and self.config['contrastive'] else None
        global_feat = self.cnn(global_feat) if global_feat is not None else None

        # reshape back to (batch_size, anatomy_size, in_features)
        node_feats = node_feats.reshape(B, N, -1)

        # for classification
        node_logits, graph_logits, graph_emb = self.model(node_feats, global_feat, return_emb=True)
        if not return_loss:
            return (node_logits, graph_logits, graph_emb) if return_emb else (node_logits, graph_logits)
        
        loss_node = self.compute_loss(node_logits, y_node)
        loss_graph = self.compute_loss(graph_logits, y_graph)
        loss = loss_graph * self.graph_importance + loss_node * (1 - self.graph_importance)

        if self.config['contrastive']:
            anchor, positive, negative = [], [], []
            y_contrastive = (torch.sum(y_node, dim=-1) > 0).int()

            for b, potential in enumerate(y_contrastive):
                positive_ids = torch.nonzero(potential == 0, as_tuple=False)
                if positive_ids.shape[0] == 0: continue
                positive_id = positive_ids[torch.randint(0, positive_ids.shape[0], (1,))]

                negative_ids = torch.nonzero(potential == 1, as_tuple=False)
                if negative_ids.shape[0] == 0: continue
                negative_id = negative_ids[torch.randint(0, negative_ids.shape[0], (1,))]

                anchor_nodes = node_feats[b]    
                anchor_global = global_images[b]
                positive_nodes = anchor_nodes.clone()
                positive_nodes[positive_id.item(), :] = 0
                positive_global = anchor_global.clone() * masks[b, positive_id.item()]
                negative_nodes = anchor_nodes.clone()
                negative_nodes[negative_id.item(), :] = 0
                negative_global = anchor_global.clone() * masks[b, negative_id.item()]

                anchor.append((anchor_nodes, anchor_global))
                positive.append((positive_nodes, positive_global))
                negative.append((negative_nodes, negative_global))
            
            anchor_nodes, anchor_global = zip(*anchor)
            anchor_nodes, anchor_global = torch.stack(anchor_nodes), torch.stack(anchor_global)
            positive_nodes, positive_global = zip(*positive)
            positive_nodes, positive_global = torch.stack(positive_nodes), torch.stack(positive_global)
            negative_nodes, negative_global = zip(*negative)
            negative_nodes, negative_global = torch.stack(negative_nodes), torch.stack(negative_global)

            anchor_global = self.cnn(anchor_global)
            positive_global = self.cnn(positive_global)
            negative_global = self.cnn(negative_global)

            _, _, anchor_emb = self.model(anchor_nodes, anchor_global, return_emb=True)
            _, _, positive_emb = self.model(positive_nodes, positive_global, return_emb=True)
            _, _, negative_emb = self.model(negative_nodes, negative_global, return_emb=True)
            anchor_emb = self.hash_fn(anchor_emb)
            positive_emb = self.hash_fn(positive_emb)
            negative_emb = self.hash_fn(negative_emb)

            loss_contrastive = self.compute_contrastive_loss(anchor_emb, positive_emb, negative_emb)
            loss = loss + loss_contrastive * self.graph_importance
        
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
