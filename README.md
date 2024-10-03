# Named Entity Recognition - Labeler Language Model

Create NER dataset fast and simple with specialized NER-LLM.

## Train Instalation

1. CUDA Installation
If you do not hae CUDA install you can visit my [tutorial](https://github.com/milistu/cuda-cudnn-installation) on how to quickly set it up.

2. Create and activate virtual environement
```bash
python -m venv .venv
```
```bash
source .venv/bin/activate
```
3. Library Installation
Update pip:
```bash
pip install --upgrade pip
```
Install PyTorch:
```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```
Install missing packages:
```bash
pip install packaging wheel
```
Install Unsloth:
```bash
pip install "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"
```
Install Jupyter:
```bash
pip install jupyter
```