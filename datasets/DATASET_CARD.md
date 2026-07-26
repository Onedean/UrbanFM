---
license: mit
task_categories:
  - time-series-forecasting
language:
  - en
tags:
  - spatio-temporal
  - urban-computing
  - foundation-model
size_categories:
  - 1B<n<10B
---

# UrbanFM Datasets

Datasets for [UrbanFM: Scaling Urban Spatio-Temporal Foundation Models](https://github.com/Onedean/UrbanFM).

## Structure

- `eval_datasets/` — evaluation datasets (traffic flow, speed, occupancy, etc.)
- `full_pretrain_datasets/` — full-scale pre-training datasets
- `pretrain_datasets/` — pre-training datasets
- `clean.ipynb`, `spatial_index_generation.py`, `data_index_process.sh` — data processing utilities

## Download

```bash
pip install huggingface_hub
huggingface-cli download Onedean/UrbanFM-datasets --repo-type dataset --local-dir ./datasets
```

## Citation

If you use these datasets, please cite the UrbanFM paper and star the [GitHub repository](https://github.com/Onedean/UrbanFM).
