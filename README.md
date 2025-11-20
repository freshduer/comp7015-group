- env setup
```bash
conda create -n comp7015 python=3.10
conda activate comp7015

pip install uv

# for data process
uv pip install ipykernel numpy pandas scikit-learn matplotlib tqdm regex 

# for lstm
# uv pip install torch
uv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# for bert
uv pip install transformers datasets accelerate evaluate

# for painting
uv pip install matplotlib

```

- dataset preprocess
```bash
makdir data
cd ./data
tar -xvf aclImdb_v1.tar.gz

run imdb_preprocess.ipynb

```