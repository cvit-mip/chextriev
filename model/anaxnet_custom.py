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
    def __init__(self, num_classes, anatomy_size=18, in_channel=1024, is_global_feat=False):
        super(AnaXnetGCN, self).__init__()
        anatomy_out = 1024
        self.num_classes = num_classes

        self.anatomy_gc1 = GraphConvolution(in_channel, 2048)
        self.anatomy_gc2 = GraphConvolution(2048, anatomy_out)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=2)

        self.norm = nn.Sequential(
            nn.LayerNorm(anatomy_out),
            nn.AdaptiveAvgPool2d((anatomy_size, anatomy_out)),
        )
        self.fc_node = nn.Linear(anatomy_out, num_classes)
        self.fc_graph = nn.Linear(anatomy_out, num_classes) if not is_global_feat else nn.Linear(anatomy_out*2, num_classes)

        #anatomy adjacency matrix
        try:
            anatomy_inp_name = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/anatomy_matrix.csv'
            anatomy_inp = pd.read_csv(anatomy_inp_name, sep='\t')
        except:
            anatomy_inp_name = '/scratch/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/anatomy_matrix.csv'
            anatomy_inp = pd.read_csv(anatomy_inp_name, sep='\t')

        self.anatomy_inp_tensor = nn.Parameter(torch.FloatTensor(anatomy_inp.values))

    def anatomy_gcn(self, feature, adj=None):
        anatomy_inp = feature
        adj = self.anatomy_inp_tensor.detach() if adj is None else adj
        x = self.anatomy_gc1(anatomy_inp, adj)
        x = self.relu(x)
        x = self.anatomy_gc2(x, adj)

        # attention logic: q-feature, k-x, v-anatomy_inp
        x = x.transpose(1, 2)  # (B, C, N)
        x = torch.matmul(feature, x)  # (B, N, C) * (B, C, N) -> (B, N, N)
        x = self.softmax(x)
        x = torch.matmul(x, anatomy_inp)  # (B, N, N) * (B, N, C) -> (B, N, C)
        return x

    def forward(self, feature, global_feat=None):
        anatomy = self.anatomy_gcn(feature)
        anatomy = anatomy.add(feature)
        # modifications
        anatomy = self.norm(anatomy)
        node_level = self.fc_node(anatomy)
        anatomy = torch.mean(anatomy, dim=1)
        anatomy = torch.cat([anatomy, global_feat], dim=-1) if global_feat is not None else anatomy
        graph_level = self.fc_graph(anatomy)
        return node_level, graph_level
    
    def retrieval_pass(self, feature, global_feat=None):
        anatomy = self.anatomy_gcn(feature)
        anatomy = anatomy.add(feature)
        # modifications
        node_emb = self.norm(anatomy)
        graph_emb = torch.mean(node_emb, dim=1)
        graph_emb = torch.cat([graph_emb, global_feat], dim=-1) if global_feat is not None else graph_emb
        return graph_emb


class CustomModel(pl.LightningModule):
    def __init__(self, config, in_features=1024, hidden_features=1024, out_features=1024, num_classes=9, num_nodes=18):
        super().__init__()
        print(f'Anaxnet using global features is {config["is_global_feat"]}')
        self.lr = config['lr']
        
        # load cnn weights
        model_paths = sorted(glob(f'/home/ssd_scratch/users/arihanth.srikar/checkpoints/mimic-cxr/resnet50/*.ckpt'))
        model_criteria = 'auc'
        model_path = [md_name for md_name in model_paths if model_criteria in md_name][-1]
        model_weights = torch.load(model_path)['state_dict']
        model_weights = {k.replace('model.', ''): v for k, v in model_weights.items()}
        
        # image featurizer
        self.cnn = resnet50(ResNet50_Weights.DEFAULT)
        # self.cnn.fc = nn.Linear(2048, num_classes)
        # self.cnn.load_state_dict(model_weights)
        self.cnn.fc = nn.Linear(2048, in_features)

        # freeze weights of cnn
        for param in self.cnn.parameters():
            param.requires_grad = False
        # unfreeze the fc layer
        for param in self.cnn.fc.parameters():
            param.requires_grad = True

        self.model = AnaXnetGCN(num_classes=num_classes, anatomy_size=num_nodes, in_channel=in_features, is_global_feat=config['is_global_feat'])

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

        self.graph_importance = config['graph_importance']
        # self.graph_importance = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        self.save_hyperparameters()

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

    def forward(self, images, y_node, y_graph, global_feat=None):
        # images (batch_size, anatomy_size, 3, 224, 224)
        # y (batch_size, anatomy_size, num_classes)
        B, N, C, H, W = images.shape
        images = images.reshape(B*N, C, H, W)

        # run it through the image featurizer (B*N, in_features)
        node_feats = self.cnn(images)
        global_feat = self.cnn(global_feat) if global_feat is not None else None

        # reshape back to (batch_size, anatomy_size, in_features)
        node_feats = node_feats.reshape(B, N, -1)

        # for classification
        node_logits, graph_logits = self.model(node_feats, global_feat)
        loss_node = self.compute_loss(node_logits, y_node)
        loss_graph = self.compute_loss(graph_logits, y_graph)
        loss = loss_graph * self.graph_importance + loss_node * (1 - self.graph_importance)
        
        return node_logits, graph_logits, loss
    
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
        node_logits, graph_logits, loss = self(batch['node_feat'], y_node, y_graph, global_feat)

        self.val_epoch_end_outputs_node.append((node_logits.detach().cpu(), y_node.detach().cpu(), loss.detach().cpu()))
        self.val_epoch_end_outputs_graph.append((graph_logits.detach().cpu(), y_graph.detach().cpu()))

        return loss
    
    def test_step(self, batch, batch_idx):
        y_node = batch['y']
        y_graph = (torch.sum(y_node, dim=1) > 0).float()
        global_feat = batch['global_feat'] if self.config['is_global_feat'] else None
        node_logits, graph_logits, loss = self(batch['node_feat'], y_node, y_graph, global_feat)

        self.test_epoch_end_outputs_node.append((node_logits.detach().cpu(), y_node.detach().cpu(), loss.detach().cpu()))
        self.test_epoch_end_outputs_graph.append((graph_logits.detach().cpu(), y_graph.detach().cpu()))

        return loss
    
    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
    
        node_logits, node_ys, node_losses = zip(*self.val_epoch_end_outputs_node)
        node_logits = torch.cat(node_logits)
        node_y = torch.cat(node_ys)
        loss = torch.mean(torch.tensor(list(node_losses)))
        
        graph_logits, graph_ys = zip(*self.val_epoch_end_outputs_graph)
        graph_logits = torch.cat(graph_logits)
        graph_y = torch.cat(graph_ys)

        self.common_end_epoch_function(node_logits, node_y, loss, graph_logits, graph_y, 'val')
    
        self.val_epoch_end_outputs_node = []
        self.val_epoch_end_outputs_graph = []
    
    @torch.no_grad()
    def on_test_epoch_end(self):

        node_logits, node_ys, node_losses = zip(*self.test_epoch_end_outputs_node)
        node_logits = torch.cat(node_logits)
        node_y = torch.cat(node_ys)
        loss = torch.mean(torch.tensor(list(node_losses)))
        
        graph_logits, graph_ys = zip(*self.test_epoch_end_outputs_graph)
        graph_logits = torch.cat(graph_logits)
        graph_y = torch.cat(graph_ys)

        self.common_end_epoch_function(node_logits, node_y, loss, graph_logits, graph_y, 'test')
    
        self.test_epoch_end_outputs_node = []
        self.test_epoch_end_outputs_graph = []
    
    
    @torch.no_grad()
    def common_end_epoch_function(self, node_logits, node_y, loss, graph_logits, graph_y, split):

        average_metrics = {
            'acc': [],
            'auc': [],
            'f1': [],
            'precision': [],
            'recall': [],
        }
        
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
    def retrieval_pass(self, images, global_img=None):
        # images (batch_size, anatomy_size, 3, 224, 224)
        # y (batch_size, anatomy_size, num_classes)
        B, N, C, H, W = images.shape
        images = images.reshape(B*N, C, H, W)

        # run it through the image featurizer (B*N, in_features)
        node_feats = self.cnn(images)
        global_feat = self.cnn(global_img) if global_img is not None else None

        # reshape back to (batch_size, anatomy_size, in_features)
        node_feats = node_feats.reshape(B, N, -1)

        # retieval
        graph_emb = self.model.retrieval_pass(node_feats)
        graph_emb = torch.cat([graph_emb, global_feat], dim=1) if global_feat is not None else graph_emb
        
        return graph_emb
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }
