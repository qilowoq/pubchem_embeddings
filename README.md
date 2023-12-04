This repo contains code for training the MXMNet and PAMNet models to predict result of biological assays from PCBA dataset. The models are trained on the 3D conformers of the molecules. Models also can be used to get molecular embeddings.

```bash
pip install -r requirements.txt
```

Then run `train_model.ipynb` to train the model.

## Data

Downloads automatically from PubChem.

## Conformers

Generated using RDKit.

## Models

### MXMNet

```
@article{zhang2020molecular,
  title={Molecular mechanics-driven graph neural network with multiplex graph for molecular structures},
  author={Zhang, Shuo and Liu, Yang and Xie, Lei},
  journal={arXiv preprint arXiv:2011.07457},
  year={2020}
}
```

### PAMNet

```
@article{zhang2023universal,
  title={A Universal Framework for Accurate and Efficient Geometric Deep Learning of Molecular Systems},
  author={Zhang, Shuo and Liu, Yang and Xie, Lei},
  journal={Scientific Reports},
  volume={13},
  number={1},
  pages={19171},
  year={2023},
  publisher={Nature Publishing Group UK London}
}

@article{zhang2022physics,
  title={Physics-aware graph neural network for accurate RNA 3D structure prediction},
  author={Zhang, Shuo and Liu, Yang and Xie, Lei},
  journal={arXiv preprint arXiv:2210.16392},
  year={2022}
}
```