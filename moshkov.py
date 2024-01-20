import pandas as pd
from typing import Callable, List, Optional
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, download_url
from smiles_to_graph import graph_data_from_smiles_and_labels
import torch
import numpy as np

class MoshkovCS(InMemoryDataset):
    raw_url = 'https://raw.githubusercontent.com/CaicedoLab/2023_Moshkov_NatComm/main/assay_data/assay_matrix_discrete_270_assays.csv'
    split_url = 'https://github.com/CaicedoLab/2023_Moshkov_NatComm/raw/main/splitting/scaffold_based_split.npz'


    """
    Dataset from https://www.nature.com/articles/s41467-023-37570-1 paper. 
    Contains 270 assays and 16170 molecules.

    Args:

        root (string): Root directory where the dataset should be saved.

        with_hydrogen (bool): If True, then hydrogen atoms will be added to the graph.
        False by default.

        with_coords (bool): If True, then coordinates of atoms will be added to the graph.
        True by default.

        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version.

        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version.

        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, 
        indicating whether the data object should be included in the final dataset.

    """
    def __init__(
        self,
        root: str,
        with_hydrogen: bool = False,
        with_coords: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ) -> None:
        self.with_hydrogen = with_hydrogen
        self.with_coords = with_coords
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> List[str]:
        return ['assay_matrix_discrete_270_assays.csv', 'scaffold_based_split.npz']
    
    @property
    def processed_file_names(self) -> str:
        return 'data_mashkov_cs.pt'
    
    def download(self):
        download_url(self.raw_url, self.raw_dir)
        download_url(self.split_url, self.raw_dir)

    def process(self):
        ds = pd.read_csv(self.raw_paths[0])
        ys = ds[ds.columns[1:]].to_numpy()
        smiles = ds['smiles'].to_list()
        data_list = []
        for i, smile in tqdm(enumerate(smiles), total=len(smiles)):
            data = graph_data_from_smiles_and_labels(smile, with_hydrogen=self.with_hydrogen, with_coords=self.with_coords)
            if data is None:
                continue
            data.y = torch.tensor(ys[i], dtype=torch.float).view(1, -1)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        self.save(data_list, self.processed_paths[0])

    def split(self, index: int = 0):
        assert index in list(range(5)) # 5-fold cross-validation
        all_ids = np.arange(len(self))
        test_ids = np.load(self.raw_paths[1])['features'][index]
        train_ids = np.setdiff1d(all_ids, test_ids)
        train_ds = self[torch.tensor(train_ids)]
        test_ds = self[torch.tensor(test_ids)]
        return train_ds, test_ds


if __name__ == "__main__":
    ds = MoshkovCS('data/moshkov_cs', with_hydrogen=False, with_coords=True)