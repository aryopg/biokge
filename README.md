# KGE Playground


## Setup
### Installation

```
conda update conda
conda env create -f environment.yml
```

### WandB API Key

If you have a WandB account:
1. Ask for an invitation from Aryo to the project page;
2. Check [your account settings](https://wandb.ai/settings) and copy your API key.

```
mkdir env
echo "WANDB_API_KEY=<YOUR WANDB API KEY>" > env/.env
```

## Example Trainings
### OGBL-PPA
```
conda activate kge_playground
sh train_local.sh ogbl-ppa
```

### BioKG

You must register to drugbank before using BioKG.
Once you received the username and password, request for a data access approval via [this link](https://www.drugbank.ca/releases/latest/downloads/all-full-database) which may take up to 2 business days.
After you received the access approval, you can start to run BioKG:

```
# Build BioKG locally, ideally in a separate folder
cd ..
git clone https://github.com/dsi-bdi/biokg.git
cd biokg
conda create -n biokg
conda install --yes --file requirements.txt
conda activate biokg
python run_all.py '<drugbank_username>' '<drugbank_password>'
conda deactivate


# Go to the kge-playground folder and start training
cd ../kge-playground
conda activate kge_playground
sh train_local.sh dsi-bdi-biokg
```


## Datasets

### General Statistics

**Note: OGBL-BioKG does not provide a clear description of how the data was developed**

| Name | Citation | Downstream tasks | Node types | Edge types |
| --- | --- | --- | --- | --- |
| BioKG | Walsh et al., 2019 | Link Property Prediction | - protein<br>- drugs<br>- indications<br>- diseases<br>- gene ontology<br>- expressions<br>- pathways | - protein–protein<br>- drug–protein<br>- drug–drug<br>- protein-genetic disorders<br>- protein-diseases<br>- protein–pathway<br>- disease–genetic disorder<br>- disease–pathway<br>- drug–pathway<br>- complex-pathway | 
| OGBL-BioKG | Hu et al., 2020 | Link Property Prediction | - diseases<br>- proteins<br>- drugs<br>- side effects<br>- protein functions | - drug-drug<br>- protein-protein<br>- drug-protein<br>- drug-side effect<br>- drug-protein function<br>- function-function |
| OGBL-PPA | Hu et al., 2020 | Link Property Prediction | - protein | - protein-protein |
### KG properties description

This table acts as a guideline when choosing KGE architectures. The chosen architectures need to be able to treat the KG properties.

| Name | #Entities | #Relation Types | #Triples | Directed | Symmetry | Inverse | Transitive | 1-to-N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BioKG | 105524 | 17 | 2067997 | Yes | Partial | Yes | Partial | Yes |
| OGBL-BioKG | 45085 | 51 | 5088433 | Yes | Partial | Yes | Partial | Yes |
| OGBL-PPA | 576289 | 1 | 30326273 | No | Yes | No | Partial | Yes |