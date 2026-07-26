# UrbanFM Datasets

The full dataset (~10 GB) is hosted on Hugging Face:

**https://huggingface.co/datasets/Onedean/UrbanFM-datasets**

## Download

Install the Hugging Face CLI:

```bash
pip install huggingface_hub
```

Download the entire datasets folder into the project root:

```bash
huggingface-cli download Onedean/UrbanFM-datasets --repo-type dataset --local-dir ./datasets
```

Or use the provided script from the project root:

```bash
bash scripts/download_datasets.sh
```

## Contents

- `eval_datasets/` — evaluation datasets
- `full_pretrain_datasets/` — full pre-training datasets
- `pretrain_datasets/` — pre-training datasets
- `clean.ipynb`, `spatial_index_generation.py`, `data_index_process.sh` — data processing scripts
