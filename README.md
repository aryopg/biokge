# Knowledge Graph Embeddings in the Biomedical Domain: Are They Useful? A Look at Link Prediction, Rule Learning, and Downstream Polypharmacy Tasks
<!-- A group project in the context of the [CDT in Biomedical AI](https://web.inf.ed.ac.uk/cdt/biomedical-ai), at the [University of Edinburgh](https://www.ed.ac.uk/). -->
This repository contains code for training and finetuning for "Knowledge Graph Embeddings in the Biomedical Domain: Are They Useful? A Look at Link Prediction, Rule Learning, and Downstream Polypharmacy Tasks" (in submission)
Authors: [Aryo Pradipta Gema](https://aryopg.github.io/), [Dominik Grabarzcyk](https://www.linkedin.com/in/dominik-grabarczyk/), [Wolf De Wulf](https://wolfdewulf.eu)   
Supervisors: [Piyush Borole](https://www.linkedin.com/in/piyush-borole/) [Dr. Javier Alfaro](https://www.proteogenomics.ca/), [Dr. Pasquale Minervini](https://neuralnoise.com/), [Dr. Antonio Vergari](http://nolovedeeplearning.com/), [Dr. Ajitha Rajan](https://homepages.inf.ed.ac.uk/arajan/) 


## 1. Installation

Create an [anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment using the [environment.yaml](./environment.yaml) file:
```
conda env create -f environment.yml
```

Activate the environment:
```
conda activate kge
```

Clone and install [libkge](https://github.com/uma-pi1/kge):
```
git clone git@github.com:uma-pi1/kge.git
cd libkge
pip install -e .
```

To deactivate the environment:
```
conda deactivate
```

## 2. Knowledge Graphs

The used knowledge graphs are those from the [v1.0.0 release of BIOKG](https://github.com/dsi-bdi/biokg/releases/tag/v1.0.0):
- BIOKG
- BIOKG benchmarks:
  * ddi_efficacy
  * ddi_minerals
  * dpi_fda
  * dep_fda_exp

Download them using the [download.py](scripts/data/download.py) script:
```
python scripts/data/download.py --help
```
The set seed ensures that they are the same as the ones used in our evaluations.
We can also provide them upon request.

The libkge dataset format is used.
Once downloaded, dataset folders need to be moved to ``kge/data``.

## 3. Experimental Evaluations

### Link prediction
All configuration files for the link prediction evaluations mentioned in the article can be found in the [configs/link_prediction](./configs/link_prediction) folder.  
Please read through the [libkge](https://github.com/uma-pi1/kge) documentation to find out how to use them.  
To be able to run the evaluations where models are initialised with pretrained embeddings, make sure to download the ``models`` folder from the [supplementary material](https://uoe-my.sharepoint.com/:f:/g/personal/s2412861_ed_ac_uk/Eta5QmbHQndPrvyZhNPROF0BBJ1T1nXPgtlHmGjxMxbpcg?e=ZFjTQY).

**Warning:** The [HPO runs](configs/hpo) can take up to a week to finish and some of the generated configurations might require a high-end GPU to be able to run at all.
During research, these HPO runs were ran on HPC clusters.

### Relation Classification
All configuration files for the relation classification evaluations mentioned in the article can be found in the [configs/relation_classification](./configs/relation_classication) folder.  
To reproduce our results, use the [relation_classification.py](scripts/benchmarking/relation_classification.py) script in combination with one of the config files:
```
python scripts/benchmarks/relation_classification.py --help
```

## 4. Questions
Feel free to contact any of the authors via email if you have questions. 
