import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl
from torchvision.models import densenet121, DenseNet121_Weights

from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryAUROC, MultilabelAUROC

os.environ['TORCH_HOME'] = '/ssd_scratch/users/arihanth.srikar'


def find_optimal_threshold(y_scores, y_true):
    # Ensure inputs are tensors
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_scores = torch.tensor(y_scores, dtype=torch.float32)

    # Initialize variables
    thresholds = torch.unique(y_scores)
    num_thresholds = len(thresholds)
    aurocs = torch.zeros(num_thresholds)

    # Calculate Binary AUROC for each threshold
    for i, threshold in enumerate(thresholds):
        y_pred = (y_scores >= threshold).float()
        aurocs[i] = roc_auc_score(y_true.numpy(), y_pred.numpy())

    # Find the optimal threshold
    optimal_index = torch.argmax(aurocs)
    optimal_threshold = thresholds[optimal_index].item()
    max_auroc = aurocs[optimal_index].item()

    return optimal_threshold, max_auroc


class CustomModel(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.lr = config['lr']
        self.test_outputs = []

        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.model.classifier = nn.Sequential(nn.Linear(1024, config['num_classes']), nn.Sigmoid())

        self.save_hyperparameters()

        # self.model = torch.compile(self.model, disable=not config['compile'])
        self.calc_metrics = {
            'f1': BinaryF1Score(),
            'auc': BinaryAUROC(),
            'acc': BinaryAccuracy(),
            'recall': BinaryRecall(),
            'precision': BinaryPrecision(),
            'multi_auc': MultilabelAUROC(num_labels=config['num_classes']),
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
        self.loss_weights = [
                            [5.913773442808936, 1.2290837000482069],
                            [5.931797809584472, 1.2110593332726713],
                            [6.867245021034882, 0.2756121218222616],
                            [6.4598478673747355, 0.6830092754824084],
                            [6.954056256933379, 0.18880088592376332],
                            [6.999514058425278, 0.1433430844318646],
                            [6.939768413023234, 0.20308872983390863],
                            [5.704353920348174, 1.4385032225089684],
                            [4.4084387336091515, 2.734418409247991],
                            [5.695777341957667, 1.447079800899476],
                            [7.078600565705689, 0.06425657715145308],
                            [6.649384441738315, 0.49347270111882724],
                            [6.871794673228491, 0.2710624696286516],
                            [5.557448555049824, 1.5854085878073194]
                        ]

        # self.criterion = nn.BCELoss(weight=torch.tensor(self.loss_weights))
        self.criterion = nn.BCELoss()

        self.val_epoch_end_outputs = []
        self.test_epoch_end_outputs = []
    
    def compute_metrics(self, logits, y):
        logits = logits.detach().cpu()
        y = y.detach().cpu().int()
        return {
            'f1': self.calc_metrics['f1'](logits, y),
            'auc': self.calc_metrics['auc'](logits, y),
            'acc': self.calc_metrics['acc'](logits, y),
            'recall': self.calc_metrics['recall'](logits, y),
            'precision': self.calc_metrics['precision'](logits, y),
        }
    
    def compute_loss(self, logits, y):
        return self.criterion(logits, y)
        # return F.binary_cross_entropy_with_logits(logits, y)
    
    def forward(self, x, y):
        logits = self.model(x)
        loss = self.compute_loss(logits, y)
        return logits, loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self(x, y)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self(x, y)

        self.val_epoch_end_outputs.append((logits.detach().cpu(), y.detach().cpu(), loss.detach().cpu()))

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self(x, y)

        self.test_epoch_end_outputs.append((logits.detach().cpu(), y.detach().cpu(), loss.detach().cpu()))

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
    