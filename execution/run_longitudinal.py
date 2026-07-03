import os
import time
import random
from concurrent import futures

import numpy as np

from data_reader import read_dataset

from core.ldp_engine import resolve_protocols
from longitudinal.prior import build_log_prior, ALL_PRIOR_TOKENS
from metrics.LASR import asr, random_asr, rr_bound_asr
from longitudinal.simulation import (
    attack_user_chunk, estimate_population_frequencies, protocol_param_str,
)


def _prior_label(prior):
    return "No PP" if prior == "no_pp" else prior


_PP_LABELS = {
    "base_pos": "Base-Pos",
    "norm":     "Norm",
    "norm_cut": "Norm-Cut",
    "norm_mul": "Norm-Mul",
    "norm_sub": "Norm-Sub",
    "power":    "Power",
    "power_ns": "Power-NS",
}

_PP_COLORS = {
    "base_pos": "#9467bd",
    "norm":     "#1f77b4",
    "norm_cut": "#2ca02c",
    "norm_mul": "#d62728",
    "norm_sub": "#ff7f0e",
    "power":    "#8c564b",
    "power_ns": "#e377c2",
}

_PP_MARKERS = {
    "base_pos": "v",
    "norm":     "^",
    "norm_cut": "D",
    "norm_mul": "o",
    "norm_sub": "s",
    "power":    "p",
    "power_ns": "x",
}

_LEGEND_ORDER = [
    "norm",     "norm_sub",
    "norm_cut", "norm_mul",
    "base_pos", "power",
    "power_ns",
    "no_pp",
    "uniform",
]


def _prior_style(prior):

    if prior == "uniform":
        return "Uniform", "#17becf", "*", "-"
    if prior == "no_pp":
        return "No PP", "black", "", "--"
    return (
        _PP_LABELS.get(prior, prior),
        _PP_COLORS.get(prior),
        _PP_MARKERS.get(prior, "o"),
        "-",
    )


# entry_point
def run_longitudinal(args, cfg):
    start = time.time()

    protocols     = resolve_protocols(cfg["experiment"]["protocols"])
    epsilon_list  = cfg["experiment"]["epsilon"]
    repeats       = cfg["experiment"]["repeats"]
    thread_number = cfg["threading"]["num_threads"]
    dataset_path  = cfg["dataset"]["path"]

    long_cfg       = cfg.get("longitudinal", {})
    observations   = long_cfg.get("observations", [1])
    consistency    = float(long_cfg.get("consistency", 1.0))
    sample_size    = long_cfg.get("sample_size", None)
    priors         = _resolve_priors(long_cfg.get("priors", ["uniform", "no_pp"]))

    save_csv   = cfg["output"].get("save_csv", False)
    csv_dir    = cfg["output"].get("csv_dir", "results")
    save_plots = cfg["output"].get("save_plots", False)
    plot_dir   = cfg["output"].get("plot_dir", "plots")

    full_path    = os.path.join(os.getcwd(), dataset_path)
    dataset_name = os.path.splitext(os.path.basename(full_path))[0]

    raw_data, _ = read_dataset.read_data(full_path)
    user_values, domain_size = _remap(raw_data)
    user_values = _subsample(user_values, sample_size)
    n_users = len(user_values)
    pop_size = n_users

    _print_settings(
        epsilon_list, observations, protocols, priors, repeats, thread_number,
        dataset_name, n_users, domain_size, consistency,
    )

    # results[(protocol, prior, eps, n)] = {"asr": ...}
    results = {}

    for protocol in protocols:
        print(f"\n{'=' * 70}")
        print(f"Protocol: {protocol.upper()}")
        print(f"{'=' * 70}")

        for epsilon in epsilon_list:
            note = protocol_param_str(protocol, epsilon, domain_size)
            note = f"  ({note})" if note else ""
            print(f"\n  --- epsilon = {epsilon}{note} ---")

            log_priors = _build_log_priors(
                priors, user_values, protocol, epsilon, domain_size, pop_size,
            )

            for n in observations:
                for prior in priors:
                    asr_val = _run_population(
                        protocol, epsilon, n, consistency,
                        user_values, domain_size,
                        log_priors[prior], thread_number, repeats,
                    )
                    results[(protocol, prior, epsilon, n)] = {"asr": asr_val}

                line = f"    n={n:>3} :"
                for prior in priors:
                    res = results[(protocol, prior, epsilon, n)]
                    line += f" [{_prior_label(prior)}] LASR={res['asr']:.4f}"
                print(line)


    _print_metric_table(results, protocols, priors, epsilon_list, observations, "asr", "LASR")

    _print_baseline_comparison(
        results, protocols, priors, epsilon_list, observations, domain_size,
    )

    if save_csv:
        _save_csv(
            results, protocols, priors, epsilon_list, observations,
            domain_size, csv_dir, dataset_name,
        )

    if save_plots:
        _plot_metric_vs_n(results, protocols, priors, epsilon_list, observations,
                          "asr", "LASR", dataset_name, plot_dir)

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"Total execution time: {elapsed:.2f} seconds")
    print(f"{'=' * 70}")


def _build_log_priors(priors, user_values, protocol, epsilon, domain_size, pop_size):
    needs_estimate = any(p != "uniform" for p in priors)

    fhat = None
    if needs_estimate:
        fhat = estimate_population_frequencies(
            user_values, protocol, epsilon, domain_size,
            seed=random.randint(1, 100_000_000),
        )

    log_priors = {}
    for prior in priors:
        log_priors[prior] = build_log_prior(
            fhat, prior, protocol, epsilon, domain_size, pop_size,
        )
    return log_priors


def _resolve_priors(requested):
    resolved = []
    for p in requested:
        p = p.lower()
        if p not in ALL_PRIOR_TOKENS and p not in ("none", "raw"):
            raise ValueError(
                f"Unknown prior '{p}'. Available: {ALL_PRIOR_TOKENS} "
                "(plus aliases 'none'/'raw' for 'no_pp')."
            )
        resolved.append("no_pp" if p in ("none", "raw") else p)
    seen = set()
    return [p for p in resolved if not (p in seen or seen.add(p))]


def _run_population(
    protocol, epsilon, n, consistency,
    user_values, domain_size,
    log_prior, thread_number, repeats,
):
    all_labels = []
    all_preds = []

    for _ in range(repeats):
        chunks = _split(user_values, thread_number)
        seeds = [random.randint(1, 100_000_000) for _ in chunks]

        tasks = [
            (
                chunk, protocol, epsilon, n, consistency,
                domain_size, log_prior, seed,
            )
            for chunk, seed in zip(chunks, seeds)
        ]

        with futures.ProcessPoolExecutor(max_workers=thread_number) as executor:
            for labels, preds in executor.map(attack_user_chunk, tasks):
                all_labels.extend(labels)
                all_preds.extend(preds)

    return asr(all_labels, all_preds)


def _remap(raw_data):
    unique = sorted(set(raw_data))
    index_of = {v: i for i, v in enumerate(unique)}
    return [index_of[v] for v in raw_data], len(unique)


def _subsample(user_values, sample_size):
    if not sample_size or sample_size >= len(user_values):
        return list(user_values)
    return random.sample(user_values, int(sample_size))


def _split(values, parts):
    parts = max(1, min(parts, len(values)))
    return [list(chunk) for chunk in np.array_split(values, parts)]


def _print_settings(
    epsilon_list, observations, protocols, priors, repeats, thread_number,
    dataset_name, n_users, domain_size, consistency,
):
    print("=" * 70)
    print("LONGITUDINAL BAYESIAN ATTACK")
    print("=" * 70)
    print(f"  Epsilon values   : {epsilon_list}")
    print(f"  Observations (n) : {observations}")
    print(f"  Protocols        : {protocols}")
    print(f"  Priors           : {[_prior_label(p) for p in priors]}")
    print(f"  Repeats          : {repeats}")
    print(f"  Threads          : {thread_number}")
    print(f"  Dataset          : {dataset_name}")
    print(f"  Users (sampled)  : {n_users}")
    print(f"  Domain size      : {domain_size}")
    print(f"  Consistency      : {consistency}")
    print(f"  Metric           : LASR")
    print("  Prior tokens     : uniform=no prior | No PP=estimate, no PP | <pp>=estimate + PP")
    print("=" * 70)


def _print_metric_table(results, protocols, priors, epsilon_list, observations, key, label):
    print(f"\n{'=' * 70}")
    print(f"{label} — rows: (protocol, prior, epsilon)   columns: observations n")
    print(f"{'=' * 70}")
    header = f"{'protocol / prior / eps':<30} | " + " | ".join(f"n={n:<4}" for n in observations)
    print(header)
    print("-" * len(header))
    for protocol in protocols:
        for prior in priors:
            for eps in epsilon_list:
                cells = " | ".join(
                    f"{results[(protocol, prior, eps, n)][key]:.4f}" for n in observations
                )
                row_label = f"{protocol.upper()} {_prior_label(prior)} e={eps}"
                print(f"{row_label:<30} | {cells}")


def _print_baseline_comparison(
    results, protocols, priors, epsilon_list, observations, domain_size,
):
    n = max(observations)
    print(f"\n{'=' * 70}")
    print(f"BASELINE COMPARISON  (n = {n})")
    print(f"  Random A: LASR=1/|D|   |   RR Bound: epsilon-LDP ceiling")
    print(f"{'=' * 70}")

    rand_asr = random_asr(domain_size)

    for protocol in protocols:
        print(f"\n  {protocol.upper()}")
        for eps in epsilon_list:
            rrb = rr_bound_asr(eps, domain_size)
            cells = "  ".join(
                f"{_prior_label(prior)}={results[(protocol, prior, eps, n)]['asr']:.4f}"
                for prior in priors
            )
            print(f"    LASR e={eps:<4} Random={rand_asr:.4f} RRBound={rrb:.4f} | {cells}")


def _save_csv(
    results, protocols, priors, epsilon_list, observations,
    domain_size, csv_dir, dataset_name,
):
    import csv

    os.makedirs(csv_dir, exist_ok=True)
    filename = os.path.join(csv_dir, f"{dataset_name}_longitudinal.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "protocol", "prior", "epsilon", "n_observations",
            "lasr", "random_lasr", "rr_bound_lasr",
        ])
        for protocol in protocols:
            for prior in priors:
                for eps in epsilon_list:
                    for n in observations:
                        res = results[(protocol, prior, eps, n)]
                        writer.writerow([
                            protocol, prior, eps, n,
                            f"{res['asr']:.6f}",
                            f"{random_asr(domain_size):.6f}",
                            f"{rr_bound_asr(eps, domain_size):.6f}",
                        ])
    print(f"\nResults saved to: {filename}")


def _plot_metric_vs_n(
    results, protocols, priors, epsilon_list, observations, key,
    label, dataset_name, plot_dir,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    os.makedirs(plot_dir, exist_ok=True)
 
    for protocol in protocols:
        for eps in epsilon_list:
            plt.figure(figsize=(8, 6))
            handle_by_prior = {}
            for prior in priors:
                ys = [results[(protocol, prior, eps, n)][key] for n in observations]
                lbl, color, marker, linestyle = _prior_style(prior)
                line, = plt.plot(
                    observations, ys,
                    label=lbl, color=color, marker=marker, linestyle=linestyle,
                    linewidth=4,
                    markersize=12,
                    zorder=10 if prior == "no_pp" else 2,
                )
                handle_by_prior[prior] = (line, lbl)
            plt.xlabel("Number of observations (n)", fontsize=28)
            plt.ylabel(label, fontsize=28)
            # plt.ylim(0, 1.0)
            plt.grid(True, alpha=0.9, linestyle="-")
 
            ordered = [handle_by_prior[p] for p in _LEGEND_ORDER if p in handle_by_prior]
            if ordered:
                handles, labels = zip(*ordered)
                plt.legend(
                    handles,
                    labels,
                    fontsize=21,
                    loc="best",
                    ncol=2,
                    columnspacing=1.5,
                    handletextpad=0.6,
                )
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=22)
            plt.tight_layout()
            filename = os.path.join(
                plot_dir, f"LONG_{key.upper()}_{protocol}_{dataset_name}_eps{eps}.png"
            )
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  Plot saved: {filename}")
 
