# KGE Playground

## Installation

```
conda update conda
conda create -n kge_playground python=3.9
conda activate kge_playground
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge ogb wandb
```

## Example Training
```
conda activate kge_playground
python train_ogb.py \\
--hidden_channels=500 \\
--reg_lambda=1e-3 \\
--epochs=100 \\
--batch_size=1000 \\
--runs=1
```