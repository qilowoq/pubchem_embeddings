from .pamnet import PAMNet
from torchmetrics import AUROC
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from tabulate import tabulate

def partial_label_masking(target):
    """
    Выбирает равное количество экземпляров из положительного и отрицательного классов.
    Игнорит nan.
    https://arxiv.org/abs/2105.10782 но не совсем точная реализация
    
    Args:
        target (torch.Tensor): target tensor
    """
    if target.is_cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    target = target.cpu()
    shapes = target.shape
    target = target.flatten()
    pos_mask = (target == 1)
    pos_count = pos_mask.sum().item()
    neg_probs = (target == 0).float()
    neg_probs[neg_probs == 0] = - torch.inf
    neg_probs = torch.nn.functional.softmax(neg_probs, dim=0)
    neg_mask = torch.full_like(pos_mask, False)
    neg_inds = np.random.choice(range(len(target)), size=pos_count, p=neg_probs.numpy(), replace=False)
    neg_mask[neg_inds] = True
    return (neg_mask + pos_mask).reshape(shapes).to(device)


class PAMNetModel(pl.LightningModule):
    def __init__(self, hparams):
        """
        PAMNet model for molecular property prediction and molecular representation learning.
        
        Args:
        
            hparams (dict): dictionary with model hyperparameters.
            Dictionary must contain the following keys:
                - learning_rate (float): learning rate for AdamW optimizer.
                - batch_size (int): batch size for training and validation.
                - hidden_dim (int): number of channels in the hidden layers of the model.
                - out_channels (int): number of channels in the output of the model. Size of the embedding.
                - num_layers (int): number of layers in the model.
                - cutoff (float): cutoff for the radial basis functions.
                - num_spherical (int): number of spherical harmonics.
                - num_radial (int): number of radial basis functions.
                - envelope_exponent (int): exponent for the smooth cutoff.
                - exit_dim (int): length of the y vector.
        """
        super().__init__()
        if "learning_rate" not in hparams:
            self.learning_rate = 1e-4
        else:
            self.learning_rate = hparams['learning_rate']
        if "batch_size" not in hparams:
            self.batch_size = 256
        else:
            self.batch_size = hparams['batch_size']
        if "exit_dim" not in hparams:
            self.exit_dim = 1
        else:
            self.exit_dim = hparams['exit_dim']
        self.validation_step_outputs = []
        self.clear_steps = True
        self.encoder = PAMNet(
            in_channels=hparams['in_channels'],
            hidden_dim=hparams['hidden_dim'],
            out_dim = hparams['out_channels'],
            cutoff_l = hparams['cutoff_l'],
            cutoff_g = hparams['cutoff_g'],
            n_layer = hparams['num_layers'], 
            num_spherical = hparams['num_spherical'],
            num_radial = hparams['num_radial'],
            envelope_exponent = hparams['envelope_exponent']
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hparams['out_channels'], out_features=hparams['out_channels']*2),
            nn.SiLU(),
            nn.Linear(in_features=hparams['out_channels']*2, out_features=hparams['exit_dim'])
            )
        self.crit = torch.nn.BCEWithLogitsLoss()
        self.metric = AUROC(task="binary")
        self.save_hyperparameters()
        
    def forward(self, batch):
        """
        Forward pass for MXMNet model.
        
        Args:
            
            batch (torch_geometric.data.Batch): batch of graphs.
            Batch must contain the following keys:
                - pos (torch.Tensor): node coordinates.
                - x (torch.Tensor): node features.
                - edge_index (torch.Tensor): edge indices.
                - y (torch.Tensor): target for training. Optional.
                - batch (torch.Tensor): batch indices.
        
        Returns:
        
            dict: Dictionary containing the following keys:
                - embeds (torch.Tensor): embeddings for downstream tasks.
                - predictors (torch.Tensor): logits for classification task.
                - loss (torch.Tensor): loss for training. None if 'y' is not in batch.
                - metric (torch.Tensor): metric for validation. None if 'y' is not in batch.
        """
        batch.pos = batch.pos - batch.pos.mean(0) # centering the molecule on the origin
        embeds = self.encoder(batch) # embeddings for downstream tasks
        predictors = self.classifier(embeds) # logits for classification task
        if 'y' in batch:
            target = batch['y'] # target for training
            assert predictors.shape == target.shape # check shapes
            mask = partial_label_masking(target) # mask for partial label masking
            loss = self.crit(predictors[mask], target[mask]) # loss for training
            metric = self.metric(torch.sigmoid(predictors[mask]), target[mask]) # metric for validation
        else:
            loss = None
            metric = None
        return {'embeds': embeds,
                'predictors': predictors,
                'loss': loss,
                'metric': metric}

    def training_step(self, batch, batch_idx):
        """
        Training step for PAMNet model.
        
        Args:
        
            batch (torch_geometric.data.Batch): batch of graphs.
            batch_idx (int): index of the current batch.
        
        Returns:
        
            torch.Tensor: Loss value for the current batch.
        """
        result = self(batch)
        self.log("loss", result['loss'].item(), 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True, 
                 batch_size=self.batch_size)
        return result['loss']
        
    def validation_step(self, batch, batch_idx):
        """
        Validation step for PAMNet model.
        
        Args:
        
            batch (torch_geometric.data.Batch): batch of graphs.
            batch_idx (int): index of the current batch.
        """
        result = self(batch)
        y_true = batch['y'].cpu()
        y_pred = torch.sigmoid(result['predictors']).cpu()
        self.validation_step_outputs.append({'y_true': y_true, 
                                             'y_pred': y_pred, 
                                             'loss': result['loss'].item(), 
                                             'metric': result['metric'].item()})
        self.log("val_loss", result['loss'].item(), 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True, 
                 batch_size=self.batch_size)
        self.log("val_metric", result['metric'].item(),
                 on_epoch=True,
                 logger=True,
                 batch_size=self.batch_size)

    def on_validation_epoch_end(self):
        """
        Callback function called at the end of each validation epoch.
        """
        y_true = torch.cat([x['y_true'] for x in self.validation_step_outputs], dim=0)
        y_pred = torch.cat([x['y_pred'] for x in self.validation_step_outputs], dim=0)
        losses = np.array([x['loss'] for x in self.validation_step_outputs])
        metrics = np.array([x['metric'] for x in self.validation_step_outputs])
        # for every label calculate AUROC
        aurocs = []
        for i in range(self.exit_dim):
            # not nan indices
            not_nan = ~np.isnan(y_true[:, i]).bool()
            if not_nan.sum() == 0: # sanity check failes because it can sample batch with all nan labels
                aurocs.append(0.5) 
                continue
            # calculate AUROC
            auroc = self.metric(y_pred[not_nan, i], y_true[not_nan, i])
            aurocs.append(auroc)
        # calculate number of labels with AUROC > 0.9
        aurocs = np.array(aurocs)
        num_good_labels = len(aurocs[aurocs > 0.9])
        # print results
        table = tabulate([['# assays AUROC > 0.9', num_good_labels],
                          ['val mean loss', np.mean(losses)],
                          ['val mean metric', np.mean(metrics)]],
                          headers=[' ', 'CS'])
        print(table)
        # log number of labels with AUROC > 0.9
        self.log("num_good_labels", num_good_labels, 
                 on_epoch=True, 
                 logger=True, 
                 batch_size=self.batch_size)
        # log val loss
        self.log("val_loss", np.mean(losses), 
                 on_epoch=True, 
                 logger=True, 
                 batch_size=self.batch_size)
        # log val metric
        self.log("val_metric", np.mean(metrics), 
                 on_epoch=True, 
                 logger=True, 
                 batch_size=self.batch_size)
        # clear validation step outputs
        if self.clear_steps:
            self.validation_step_outputs = []
            

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        
        Returns:
        
            torch.optim.Optimizer: Optimizer for training.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)