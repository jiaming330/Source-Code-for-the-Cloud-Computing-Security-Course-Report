# MOON Reproduction Package (Course Report)

This repository contains the materials used in my reproduction of **MOON: Model-Contrastive Federated Learning**.

## What is included

- `code/MOON_full_listing.py`: a **single-file source listing** (snapshot) generated for appendix/review purposes.
- `code/MOON_full_listing.manifest.tsv`: per-file line counts and SHA256 checksums for traceability.
- `code/plot_moon_vs_fedavg.py`: my plotting script to generate the comparison curve.
- `results/*.csv` and `results/*.png`: extracted curves and the plotted figure.

> Note: Datasets (e.g., CIFAR-10) are **not** uploaded. They are downloaded automatically by torchvision during running.

## How to reproduce the figure

From repo root:

```bash
python code/plot_moon_vs_fedavg.py
```

It will read:

- `results/moon_curve.csv`
- `results/fedavg_curve.csv`

and write:

- `results/moon_vs_fedavg.png`

## Experiment settings (this run)

- Dataset: CIFAR-10
- Partition: Dirichlet non-IID, beta=0.5
- Clients: 10
- Communication rounds: 100
- Metric: Global test accuracy per communication round

Final round (round=99):

- MOON: 0.6822
- FedAvg: 0.6670



