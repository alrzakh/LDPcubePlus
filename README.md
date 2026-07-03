# LDP³+

**Benchmarking Post-Processing Methods in Local Differential Privacy for Utility and Adversarial Robustness**

LDP³+ is a research framework for evaluating **post-processing methods** for
**Local Differential Privacy (LDP) frequency estimation**, under both clean and
adversarial conditions. It simulates an LDP protocol (perturb + aggregate),
applies various post-processing methods to the noisy frequency estimates, and
measures how those methods behave with respect to **utility**, **robustness to
poisoning**, **privacy-budget leakage**, and **rank preservation**.

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/<your-username>/LDPcubePlus.git
cd LDPcubePlus
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

`Datasets/Adult.csv` is included as a small default dataset so the tool runs
out of the box. The larger datasets (Kosarak, BMS, Porto) are **not** shipped.
Each dataset is a plain text file with **one integer per line**. See [`Datasets/README.md`](Datasets/README.md) for the
expected format and the sources of the datasets used in the paper.

Point the runner at a dataset via `--dataset PATH` or via `dataset.path` in a config.

## Usage

LDP³+ is a single CLI (`main.py`) with five subcommands, each backed by its own
YAML config in `configs/`. CLI flags override the config file.

| Subcommand     | Purpose |
|----------------|---------|
| `estimate`     | Frequency estimation on clean data; measure post-processing **error** (utility). |
| `poisoning`    | **MGA poisoning attack**; measure **Gain Reduction** across an epsilon sweep. |
| `bia`          | **Budget Inference Attack**; infer the true ε via binary search. |
| `rank`         | **Rank preservation**; Kendall-Tau correlation across an ε sweep. |
| `longitudinal` | **Longitudinal Bayesian attack**; infer users' true values from repeated LDP reports (reports LASR). |

### Examples

```bash
# Frequency estimation
python main.py estimate --config configs/estimation.yaml

# Quick single-protocol test without editing the config
python main.py estimate --config configs/estimation.yaml \
    --protocol grr --epsilon 1.0 --repeat 5

# MGA poisoning attack sweep
python main.py poisoning --config configs/mga.yaml --num-malicious 1000 --save-plots

# Budget inference with binary search
python main.py bia --config configs/bia.yaml --run-bia --bia-runs 5

# Rank preservation
python main.py rank --config configs/rank.yaml

# Longitudinal Bayesian attack
python main.py longitudinal --config configs/longitudinal.yaml
```

## Supported protocols & methods

- **LDP protocols:** GRR, RAPPOR, OLH, BLH, Subset selection, OUE.
- **Post-processing methods:** `base_pos`, `norm`, `norm_cut`, `norm_mul`,
  `norm_sub`, `power`, `power_ns`.
- **Attacks:** MGA poisoning (Cao et al. 2021), Budget Inference Attack(Balioglu et al. 2026),
  Longitudinal Bayesian attack (Gürsoy 2024).

## Project layout

```
main.py            CLI entry point (argparse subcommands)
configs/           base.yaml + one YAML per subcommand
core/              perturb/aggregate orchestration + post-processing dispatch
execution/         config loader + one run_*.py per subcommand
protocols/         LDP protocols (perturb/aggregate) + analytic variances
post_processing/   post-processing methods
malicious_population/  MGA poison report crafters
longitudinal/      longitudinal Bayesian attack
metrics/           utility & attack metrics (L1, L2, KL, EMD, gain, LASR, ...)
utils/             tabulation, CSV, win-counts
data_reader/       dataset reader
Datasets/          (see Datasets/README.md)
```

## Citation

If you use this framework, please cite the accompanying paper:

> A. Khodaie, B. Balioglu, M. E. Gürsoy. *Benchmarking Post-Processing Methods in Local Differential Privacy for Utility and Adversarial Robustness.*

## License

Released under the [MIT License](LICENSE).
