- env
```bash
conda create -n comp7015 python=3.10
conda activate comp7015

pip install uv

# for data process
uv pip install ipykernel numpy pandas scikit-learn matplotlib tqdm regex 

# for lstm
uv pip install torch

# for bert
uv pip install transformers datasets accelerate evaluate

# for painting
uv pip install matplotlib

```

- dataset
```bash
makdir data
cd ./data
tar -xvf aclImdb_v1.tar.gz

run imdb_preprocess.ipynb

```