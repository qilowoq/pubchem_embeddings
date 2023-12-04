# import packages

# general tools
import numpy as np
from tqdm import tqdm
import os
from typing import Union
from typing import Optional
# RDkit
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# Pytorch and Pytorch Geometric
import torch
from torch_geometric.data import Data

RDLogger.DisableLog('rdApp.*')

def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding


def get_atom_features(atom, use_chirality = True, hydrogens_implicit = True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """
    # define list of permitted atoms
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe',
                                'As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co',
                                'Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr',
                                'Cr','Pt','Hg','Pb','Unknown']
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    # compute atom features
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])  
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled                          
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
    return np.array(atom_feature_vector)


def get_bond_features(bond, use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc
    return np.array(bond_feature_vector)


def get_positions(mol) -> Union[np.array, None]:
    """
    Takes an RDKit mol object as input and gives a 2d-numpy array of atom positions or None if getting coords is imposible.
    """
    config = AllChem.EmbedMolecule(mol, maxAttempts=100)
    if config == -1:
        return None
    AllChem.UFFOptimizeMolecule(mol)
    return mol.GetConformers()[config].GetPositions()


def graph_data_from_mol_and_labels(mol, 
                                   y: Union[np.array, list, None]= None, 
                                   smiles: Optional[str] = None,
                                   with_coords: bool = False,
                                   with_hydrogen: bool = False) -> Optional[Data]:
    """
    Inputs:
    
    mol: rdKit molecule object
    y (optional): labels for the SMILES string (such as associated pKi values) 

    Outputs:

    data: torch_geometric.data.Data object which represent labeled molecular graph that can readily be used for machine learning
    """
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)
    # get feature dimensions
    n_nodes = mol.GetNumAtoms()
    n_edges = 2*mol.GetNumBonds()
    n_node_features, n_edge_features = (79, 10)
    # !!! if breaks, use this :
    #    unrelated_mol = Chem.MolFromSmiles("O=O")
    #    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
    #    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
    # construct node feature matrix X of shape (n_nodes, n_node_features) and charge vector (n_nodes)
    X = np.zeros((n_nodes, n_node_features))
    atomic_number = np.zeros(n_nodes)
    for atom in mol.GetAtoms():
        X[atom.GetIdx(), :] = get_atom_features(atom)
        atomic_number[atom.GetIdx()] = atom.GetAtomicNum()
    z = torch.tensor(atomic_number, dtype=torch.long)
    X = torch.tensor(X, dtype = torch.float32)
    # construct edge index array E of shape (2, n_edges)
    (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
    torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
    E = torch.stack([torch_rows, torch_cols], dim = 0)
    # construct edge feature array EF of shape (n_edges, n_edge_features)
    EF = np.zeros((n_edges, n_edge_features))
    for (k, (i,j)) in enumerate(zip(rows, cols)):
        EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
    EF = torch.tensor(EF, dtype = torch.float)
    # construct Pytorch Geometric data object
    #data = Data(z=z, edge_index = E, edge_attr = EF)
    data = Data(x=X, z=z, edge_index = E, edge_attr = EF)
    if y is not None:
        # construct label tensor and add to data object
        data.y = torch.tensor(np.array([y]), dtype = torch.float32)
    if smiles is None:
        smiles = Chem.MolToSmiles(mol)
    # add smiles to data object
    data.smiles = smiles
    # construct position matrix POS of shape (n_nodes, 3)
    if with_coords:
        coords = get_positions(mol)
        if coords is None:
            return None
        data.pos = torch.tensor(coords, dtype=torch.float32)
    return data

def graph_data_from_smiles_and_labels(smiles: str, 
                                      with_hydrogen: bool = False,
                                      with_coords: bool = False,
                                      y: Union[np.array, list, None] = None) -> Optional[Data]:
    """
    Inputs:
    
    x_smiles: a  SMILES string
    y (optional): : labels for the SMILES string (such as associated pKi values)

    Outputs:

    data: a torch_geometric.data.Data object which represent labeled molecular graph that can readily be used for machine learning
    """
    # convert SMILES to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)
    return graph_data_from_mol_and_labels(mol=mol, 
                                          y=y, 
                                          smiles=smiles, 
                                          with_coords=with_coords, 
                                          with_hydrogen=with_hydrogen)