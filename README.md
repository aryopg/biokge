# Evaluating Knowledge Graph Embeddings for Protein Function
A group project in the context of the [CDT in Biomedical AI](https://web.inf.ed.ac.uk/cdt/biomedical-ai), at the [University of Edinburgh](https://www.ed.ac.uk/).  
Authors: [Aryo Pradipta Gema](https://aryopg.github.io/), Dominik Grabarzcyk, [Wolf De Wulf](https://wolfdewulf.eu)   
Supervisors: [Dr. Javier Alfaro](https://www.proteogenomics.ca/), [Dr. Pasquale Minervini](https://neuralnoise.com/), [Dr. Antonio Vergari](http://nolovedeeplearning.com/), [Dr. Ajitha Rajan](https://homepages.inf.ed.ac.uk/arajan/) 


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

All of these can be downloaded using the [download.py](scripts/data/download.py) script:
```
python scripts/data/download.py --help
```

The libkge dataset format is used.
Once downloaded, dataset folders need to be moved to ``kge/data``.

## 3. Experimental evaluations

### Link prediction
All configuration files for the evaluations mentioned in the article can be found in the [configs](./configs/) folder.
Please read through the [libkge](https://github.com/uma-pi1/kge) documentation to find out how to use them.

**Warning:** The [HPO runs](configs/hpo) can take up to a week to finish and some of the generated configurations might require a high-end GPU to be able to run at all.
During research, these HPO runs were ran on HPC clusters.

### Classification benchmarks
Evaluating a link prediction model on one of the benchmarks can be done using the [evaluate_benchmark.py](scripts/results/evaluate_benchmark.py) script:
```
python scripts/results/evaluate_benchmark.py --help
```

Evaluating the baseline models can be done using the []() script:
```
python scripts/results/evaluate_benchmark.py --help
```

## 4. Questions
Feel free to contact any of the authors via email if you have questions. 