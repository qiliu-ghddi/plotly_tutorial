[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/qiliu-ghddi/plotly_tutorial/HEAD)
# Plotly Tutorial

# Installation 

```
conda info --envs
conda create -n "plotly" python=3.6
# conda update -n base -c defaults conda
conda activate plotly
# conda deactivate

conda install -c plotly plotly=5.9.0
conda install "jupyterlab>=3" "ipywidgets>=7.6"
conda install "notebook>=5.3" "ipywidgets>=7.5"
conda install -c conda-forge python-kaleido

# https://anaconda.org/conda-forge/rdkit
conda install -c conda-forge rdkit
pip install molplotly

```