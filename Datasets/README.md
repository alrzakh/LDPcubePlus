# Datasets

This folder contains the datasets used in the paper.

## Expected format
Each dataset is a plain text file with **one integer per line**.

## Datasets used in the paper
| File          | Source |
|---------------|--------|
| `Adult.csv`   | UCI Adult dataset (age attribute). |
| `Kosarak.csv` | Kosarak click-stream dataset ([SPMF Dataset Repository](https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php)). |
| `BMS.csv`     | BMS-POS retail dataset ([Github repository](https://github.com/cpearce/HARM/blob/master/datasets/BMS-POS.csv)). |
| `porto.csv`   | Porto taxi-trajectory dataset. |

We provide the processed versions of the above-referenced datasets in this
folder to enable the reproducibility of the paper's results. Point the runner at
a file with `--dataset PATH` or via `dataset.path` in the config.
