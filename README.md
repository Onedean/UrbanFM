# UrbanFM

> Implementation for paper: UrbanFM: Scaling Urban Spatio-Temporal Foundation Models

## Datasets

The full datasets (~10 GB) are hosted on Hugging Face. Download before running experiments:

```bash
bash scripts/download_datasets.sh
```

See [datasets/README.md](datasets/README.md) for details.

## Experiments
For the default configuration, you can quickly experiment with the following scripts:

```
sh scripts/expert_forecasting/staeformer_forecasting_few.sh
```

or use following command:

```
python foundation_model_run.py
```
