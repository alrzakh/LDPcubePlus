import os
import time
from collections import Counter
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

from data_reader import read_dataset

from core.ldp_engine import run_ldp_clean, resolve_protocols
from core.postprocessor import postprocess, resolve_methods


_LABEL_MAP = {
    "base_pos": "Base-Pos",
    "norm":     "Norm",
    "norm_cut": "Norm-Cut",
    "norm_mul": "Norm-Mul",
    "norm_sub": "Norm-Sub",
    "power":    "Power",
    "power_ns": "Power-NS",
}


def run_rank(args, cfg):

    start = time.time()

    protocols      = resolve_protocols(cfg["experiment"]["protocols"])
    methods        = resolve_methods(cfg["experiment"]["methods"])
    epsilon_values = cfg["experiment"]["epsilon"]
    repeats        = cfg["experiment"]["repeats"]
    thread_number  = cfg["threading"]["num_threads"]
    dataset_path   = cfg["dataset"]["path"]

    rank_cfg        = cfg.get("rank", {})
    generate_plots  = rank_cfg.get("generate_plots", False)
    plot_dir        = cfg["output"].get("plot_dir", "plots")

    full_path    = os.path.join(os.getcwd(), dataset_path)
    dataset_name = os.path.splitext(os.path.basename(full_path))[0]

    original_data, domain_size = read_dataset.read_data(full_path)
    n      = len(original_data)
    counts = Counter(original_data)
    frequency = [counts.get(i, 0) / n for i in range(domain_size)]

    _print_settings(
        epsilon_values, protocols, methods, repeats,
        thread_number, dataset_name, n, domain_size,
        generate_plots,
    )

    results = {}

    for i in range(repeats):
        for p in protocols:
            print(f"  Protocol {p.upper()}, repeat {i + 1}/{repeats}")

            ldp_results = run_ldp_clean(
                protocol=p,
                epsilon=epsilon_values,
                thread_number=thread_number,
                input_data=deepcopy(original_data),
                domain_size=domain_size,
            )

            for eps in epsilon_values:
                est_counts = ldp_results[eps]["counts"]
                est_normal = ldp_results[eps]["normalized"]

                for m in methods:
                    post_freq = postprocess(
                        method=m,
                        normalized_original=frequency,
                        estimated_counts=est_counts,
                        estimated_counts_normalized=est_normal,
                        original_data=original_data,
                        epsilon=eps,
                        protocol=p,
                        metric=None,
                    )

                    corr = _rank_correlation(frequency, post_freq)
                    results[(p, m, eps, i)] = corr

    elapsed = time.time() - start
    print(f"\n  Experiment time: {elapsed:.2f}s")

    _print_results(results, protocols, methods, epsilon_values, repeats)

    if generate_plots:
        _plot_rank_correlations(
            results, protocols, methods, epsilon_values,
            repeats, dataset_name, plot_dir,
        )

    print(f"\n{'=' * 60}")
    print(f"Total execution time: {elapsed:.2f} seconds")
    print(f"{'=' * 60}")



def _rank_correlation(original_freq, processed_freq):
    corr, _ = kendalltau(original_freq, processed_freq)
    return float(corr)



def _print_results(results, protocols, methods, epsilon_values, repeats):

    avg  = {}
    std  = {}
    for p in protocols:
        for m in methods:
            for eps in epsilon_values:
                vals = [results[(p, m, eps, i)] for i in range(repeats)]
                avg[(p, m, eps)]  = float(np.mean(vals))
                std[(p, m, eps)]  = float(np.std(vals))

    print(f"\n{'=' * 60}")
    print(f"RESULTS: Kendall Tau Rank Correlation")
    print(f"{'=' * 60}")

    for p in protocols:
        print(f"\n  Protocol: {p.upper()}")
        print(f"  {'-' * 56}")

        for eps in epsilon_values:
            print(f"\n    ε = {eps}")
            print(f"    {'Method':<15}  {'Avg':>8}  {'Std':>8}")
            print(f"    {'-' * 35}")

            method_scores = []
            for m in methods:
                a = avg[(p, m, eps)]
                s = std[(p, m, eps)]
                label = _LABEL_MAP.get(m, m)
                print(f"    {label:<15}  {a:>8.4f}  {s:>8.4f}")
                method_scores.append((m, a))

            best_m, best_a = max(method_scores, key=lambda x: x[1])
            print(f"\n    Best: {_LABEL_MAP.get(best_m, best_m)} ({best_a:.4f})")

    win_pcts = _calculate_win_percentages(
        results, protocols, methods, epsilon_values, repeats
    )

    print(f"\n{'=' * 60}")
    print(f"WIN PERCENTAGES: Kendall Tau")
    print(f"(out of {repeats} runs per protocol, per ε)")
    print(f"{'=' * 60}")

    for eps in epsilon_values:
        print(f"\n  ε = {eps}")
        for p in protocols:
            print(f"    {p.upper()}")
            sorted_pcts = sorted(
                win_pcts[(eps, p)].items(), key=lambda x: x[1], reverse=True
            )
            for m, pct in sorted_pcts:
                if pct > 0:
                    print(f"      {_LABEL_MAP.get(m, m):<15}: {pct:5.1f}%")

    print(f"\n{'=' * 60}")
    print(f"OVERALL BEST METHOD PER PROTOCOL: Kendall Tau")
    print(f"(average correlation across all ε values)")
    print(f"{'=' * 60}")

    for p in protocols:
        method_totals = []
        for m in methods:
            total = sum(avg[(p, m, eps)] for eps in epsilon_values)
            method_totals.append((m, total / len(epsilon_values)))

        best_m, best_avg = max(method_totals, key=lambda x: x[1])
        print(
            f"  {p.upper():<10} → "
            f"{_LABEL_MAP.get(best_m, best_m):<15} "
            f"(avg corr: {best_avg:.4f})"
        )


def _calculate_win_percentages(
    results, protocols, methods, epsilon_values, repeats
):

    win_pcts = {}

    for eps in epsilon_values:
        for p in protocols:
            wins = {m: 0 for m in methods}

            for i in range(repeats):
                scores = {
                    m: results[(p, m, eps, i)]
                    for m in methods
                }
                best = max(scores, key=scores.get)
                wins[best] += 1

            win_pcts[(eps, p)] = {m: (wins[m] / repeats) * 100 for m in methods}

    return win_pcts



def _plot_rank_correlations(
    results, protocols, methods, epsilon_values,
    repeats, dataset_name, plot_dir,
):

    os.makedirs(plot_dir, exist_ok=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    method_colors = {m: colors[i] for i, m in enumerate(methods)}

    for p in protocols:
        fig, ax = plt.subplots(figsize=(10, 6))

        for m in methods:
            corr_avgs = [
                float(np.mean([results[(p, m, eps, i)] for i in range(repeats)]))
                for eps in epsilon_values
            ]
            corr_stds = [
                float(np.std([results[(p, m, eps, i)] for i in range(repeats)]))
                for eps in epsilon_values
            ]

            ax.plot(
                epsilon_values, corr_avgs,
                marker="s",
                label=_LABEL_MAP.get(m, m),
                color=method_colors[m],
                linewidth=2,
                markersize=6,
            )
            ax.fill_between(
                epsilon_values,
                np.array(corr_avgs) - np.array(corr_stds),
                np.array(corr_avgs) + np.array(corr_stds),
                alpha=0.2,
                color=method_colors[m],
            )

        ax.set_xlabel(r"$\varepsilon$", fontsize=30)
        ax.set_ylabel("Kendall Tau", fontsize=12)
        ax.set_title(f"{p.upper()}", fontsize=13)
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = os.path.join(
            plot_dir, f"{dataset_name}_{p}_kendall.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"  Plot saved: {filename}")
        plt.close()


def _print_settings(
    epsilon_values, protocols, methods, repeats,
    thread_number, dataset_name, n, domain_size,
    generate_plots,
):
    print("=" * 60)
    print("RANK PRESERVATION EXPERIMENT")
    print("=" * 60)
    print(f"  Epsilon values   : {epsilon_values}")
    print(f"  Protocols        : {protocols}")
    print(f"  Methods          : {methods}")
    print(f"  Repeats          : {repeats}")
    print(f"  Threads          : {thread_number}")
    print(f"  Dataset          : {dataset_name}")
    print(f"  Dataset size     : {n}")
    print(f"  Domain size      : {domain_size}")
    print(f"  Rank metric      : Kendall Tau")
    print(f"  Generate plots   : {generate_plots}")
    print("=" * 60)
