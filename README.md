This repo contains code for training PAMNet model to predict result of biological assays from PCBA dataset or from dataset in [this paper](https://www.nature.com/articles/s41467-023-37570-1). The model are trained on the 3D conformers of the molecules. Models also can be used to get molecular embeddings.

```bash
pip install -r requirements.txt
```

Then run `train_model.ipynb` to train the model.

## Data

Downloads automatically.

## Conformers

Generated using RDKit.

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