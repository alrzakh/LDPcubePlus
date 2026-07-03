# Datasets

`Adult.csv` is included as a small default dataset. The larger dataset CSV files
are **not** included in this repository (they are excluded via `.gitignore`). To
run experiments on them, place the dataset files here in the format below.

## Expected format

Each dataset is a plain text file with **one integer per line**.

## Datasets used in the paper

| File          | Source |
|---------------|--------|
| `Adult.csv`   | UCI Adult dataset (age attribute). |
| `Kosarak.csv` | Kosarak click-stream dataset ([SPMF Dataset Repository](https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php)). |
| `BMS.csv`     | BMS-POS retail dataset ([Github repository](https://github.com/cpearce/HARM/blob/master/datasets/BMS-POS.csv)). |
| `porto.csv`   | Porto taxi-trajectory dataset |

Download the raw datasets from their original sources and preprocess them into
the one-integer-per-line format described above before running the tool. Point
the runner at a file with `--dataset PATH` or via `dataset.path` in the config.
