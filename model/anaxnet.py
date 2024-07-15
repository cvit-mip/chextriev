import numpy as np
import pandas as pd
from glob import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, densenet121, DenseNet121_Weights
from torch_geometric.nn import GCNConv
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
    def __init__(self, num_classes, anatomy_size=18, in_channel=1024):
        super(AnaXnetGCN, self).__init__()
        anatomy_out = 1024
        self.num_classes = num_classes

        self.anatomy_gc1 = GraphConvolution(in_channel, 2048)
        self.anatomy_gc2 = GraphConvolution(2048, 1024)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=2)

        self.fc = nn.Sequential(
            nn.LayerNorm(anatomy_out),
            # nn.AdaptiveAvgPool2d((anatomy_size, anatomy_out)),
            nn.AdaptiveAvgPool1d(anatomy_out),
            nn.Linear(anatomy_out, num_classes)
        )

        #anatomy adjacency matrix
        anatomy_inp_name = '/home/ssd_scratch/users/arihanth.srikar/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/anatomy_matrix.csv'
        anatomy_inp = pd.read_csv(anatomy_inp_name, sep='\t')
        self.anatomy_inp_tensor = nn.Parameter(torch.FloatTensor(anatomy_inp.values))

    def anatomy_gcn(self, feature):
        anatomy_inp = feature
        adj = self.anatomy_inp_tensor.detach()
        x = self.anatomy_gc1(anatomy_inp, adj)
        x = self.relu(x)
        x = self.anatomy_gc2(x, adj)

        x = x.transpose(1, 2)
        x = torch.matmul(feature, x)
        x = self.softmax(x)
        x = torch.matmul(x, anatomy_inp)
        return x

    def forward(self, feature):
        anatomy = self.anatomy_gcn(feature)
        anatomy = anatomy.add(feature)
        anatomy = torch.mean(anatomy, dim=1)
        anatomy = self.fc(anatomy)
        return anatomy
    
    def retrieval_pass(self, feature):
        anatomy = self.anatomy_gcn(feature)
        anatomy = anatomy.add(feature)
        anatomy = self.fc[0](anatomy)
        anatomy = self.fc[1](anatomy)
        anatomy = torch.mean(anatomy, dim=1)
        return anatomy


class CustomModel(pl.LightningModule):
    def __init__(self, config, in_features=1024, hidden_features=1024, out_features=1024, num_classes=9, num_nodes=18):
        super().__init__()
        self.lr = config['lr']
        
        self.cnn = densenet121(DenseNet121_Weights.DEFAULT)
        self.cnn.classifier = nn.Identity()
        # self.cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
        # self.cnn.fc = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, in_features),
        # )
        # freeze weights of cnn
        for param in self.cnn.parameters():
            param.requires_grad = False
        # for param in self.cnn.fc.parameters():
        #     param.requires_grad = True

        self.model = AnaXnetGCN(num_classes=num_classes, anatomy_size=num_nodes, in_channel=in_features)

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
        
        self.lables = [
            'lung opacity', 
            'pleural effusion', 
            'atelectasis', 
            'enlarged cardiac silhouette',
            'pulmonary edema/hazy opacity', 
            'pneumothorax', 
            'consolidation', 
            'fluid overload/heart failure', 
            'pneumonia']
        assert len(self.lables) == self.num_classes, f'Number of classes in config {num_classes} does not match the number of lables in the dataset {len(self.lables)}'
        
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

    def forward(self, images, y):
        # images (batch_size, anatomy_size, 3, 224, 224)
        # y (batch_size, anatomy_size, num_classes)
        B, N, C, H, W = images.shape
        images = images.reshape(B*N, C, H, W)

        # run it through the image featurizer (B*N, in_features)
        node_feats = self.cnn(images)

        # reshape back to (batch_size, anatomy_size, in_features)
        node_feats = node_feats.reshape(B, N, -1)

        # for classification
        logits = self.model(node_feats)
        loss = self.compute_loss(logits, y)
        
        return logits, loss
    
    def training_step(self, batch, batch_idx):
        logits, loss = self(batch['node_feat'], (torch.sum(batch['y'], axis=1) > 0).float())

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        logits, loss = self(batch['node_feat'], (torch.sum(batch['y'], axis=1) > 0).float())

        self.val_epoch_end_outputs.append((logits.detach().cpu(), (torch.sum(batch['y'], axis=1) > 0).float().detach().cpu(), loss.detach().cpu()))

        return loss
    
    def test_step(self, batch, batch_idx):
        logits, loss = self(batch['node_feat'], (torch.sum(batch['y'], axis=1) > 0).float())

        self.test_epoch_end_outputs.append((logits.detach().cpu(), (torch.sum(batch['y'], axis=1) > 0).float().detach().cpu(), loss.detach().cpu()))

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
    def retrieval_pass(self, images, global_feat=None, concat_global_feature=False):
        # images (batch_size, anatomy_size, 3, 224, 224)
        # y (batch_size, anatomy_size, num_classes)
        B, N, C, H, W = images.shape
        images = images.reshape(B*N, C, H, W)

        # run it through the image featurizer (B*N, in_features)
        node_feats = self.cnn(images)

        # reshape back to (batch_size, anatomy_size, in_features)
        node_feats = node_feats.reshape(B, N, -1)

        # retieval
        emb = self.model.retrieval_pass(node_feats)
        
        return emb
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]
