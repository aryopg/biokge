# KGE Playground

## Installation

```
conda update conda
conda create -n kge_playground python=3.9
conda activate kge_playground
conda install -y clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64
MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch==1.12.1 torchvision torchaudio
python -c "import torch; print(torch.__version__)"  #---> (Confirm the version is 1.12.1)
MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cpu.html
MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cpu.html
MACOSX_DEPLOYMENT_TARGET=12.3 CC=clang CXX=clang++ python -m pip --no-cache-dir install torch-geometric
conda install -y jupyterlab matplotlib seaborn networkx
ipython kernel install --user --name=conda-kge-playground
conda install -c conda-forge rdflib
```