# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/python-3-miniconda/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/devcontainers/python:3.10

# Configure conda environment
RUN pip install ipykernel ipywidgets
RUN pip install pandas seaborn scipy scikit-learn tqdm 
RUN pip install pyyaml
RUN pip install torch torchvision torchaudio
RUN git clone https://github.com/NeuralOperator/neuraloperator /tmp/neuralop
RUN pip install -e /tmp/neuralop
RUN pip install -r /tmp/neuralop/requirements.txt
RUN pip install torch-scatter