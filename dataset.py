import torch
from tqdm import tqdm
import re
from typing import Callable, Optional
from torch_geometric.datasets import MoleculeNet
from smiles_to_graph import graph_data_from_smiles_and_labels
from torch_geometric.datasets import MoleculeNet
import os

class MoleculeNetWithNewFeatures(MoleculeNet):
    """
    Modified version of MoleculeNet dataset from torch_geometric.datasets.MoleculeNet
    
    Args:
    
        root (string): Root directory where the dataset should be saved.
        
        name (string): The name of the dataset. Probably should work with all datasets from MoleculeNet.
        But it was tested only with 'PCBA' 
        
        with_hydrogen (bool): If True, then hydrogen atoms will be added to the graph.
        Off by default.
        
        with_coords (bool): If True, then coordinates of atoms will be added to the graph.
        On by default.
        
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version.
        
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version.
    
    """
    def __init__(self, root: str, 
                 name: str, 
                 with_hydrogen: bool = False,
                 with_coords: bool = True,
                 transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None, 
                 pre_filter: Optional[Callable] = None):
        self.with_hydrogen = with_hydrogen
        self.with_coords = with_coords
        self.possible_atom_nums = {6, 7, 8, 9, 15, 16, 17, 35, 53} # C, N, O, F, P, S, Cl, Br, I
        super().__init__(root, name, transform, pre_transform, pre_filter)
        
        
    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for line in tqdm(dataset):
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(',')

            smiles = line[self.names[self.name][3]]
            ys = line[self.names[self.name][4]]
            ys = ys if isinstance(ys, list) else [ys]

            ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)

            # в y присутствуют не только nan но и 1 и 0
            if (~torch.isnan(y)).long().sum().item() == 0 or y[~torch.isnan(y)].max() == 0:
                continue

            data = graph_data_from_smiles_and_labels(smiles, with_hydrogen=self.with_hydrogen, with_coords=self.with_coords)
            if data is None:
                continue
            data.y = y

            # ограничения по длине снизу
            if len(data.z.tolist()) <= 5:
                continue

            # в молекуле только возможные тяжелые атомы
            if len(set(data.z.tolist()).difference(self.possible_atom_nums)) != 0:
                continue

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == "__main__":
    ds = MoleculeNetWithNewFeatures(root='data', name='PCBA')