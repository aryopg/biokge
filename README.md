# KGE Playground

## Installation

```
conda update conda
conda env create -f environment.yml
```

## Setup

### WandB API Key

If you have a WandB account:
1. Ask for an invitation from Aryo to the project page;
2. Check [your account settings](https://wandb.ai/settings) and copy your API key.

```
mkdir env
echo "WANDB_API_KEY=<YOUR WANDB API KEY>" > env/.env
```

## Example Training
```
conda activate kge_playground
train_local.sh
```