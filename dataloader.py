import torch
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

class GraphDataModule(pl.LightningDataModule):
    def __init__(self, dataset, hparams: dict) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = hparams['batch_size']
        self.train_split = hparams['train_split']
        self.train_shuffle = hparams['train_shuffle']
        self.seed = hparams['seed']
    
    def setup(self, stage) -> None:
        if type(self.dataset) == tuple:
            assert len(self.dataset) == 2
            self.train_dataset, self.val_dataset = self.dataset
        else:
            len_train = int(self.train_split * len(self.dataset))
            len_val = len(self.dataset) - len_train
            lengths = [len_train, len_val]
            generator1 = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, 
                                                                                lengths, 
                                                                                generator=generator1)
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle, num_workers=19)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=19)