# install basic packages
pip install numpy==1.23.5 scipy==1.10.1 scikit-learn==1.2.2 safetensors==0.3.1 biopython==1.78 rdkit-pypi==2022.9.3 networkx==2.7.1 GraphRicciCurvature==0.5.3.1

# install PyTorch
conda install -y pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch

# install DGL
conda install -y -c dglteam dgl=1.1.0

# install fair-esm
pip install fair-esm==2.0.0 

# install torchdrug
# Two dependencies of torchdrug (torch-cluster and torch-scatter) cannot be found in PyPI, so we need to specify the version and the link to the wheel file
pip install torch-cluster==1.6.1+pt113cpu -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
pip install torchdrug==0.2.0.post1

