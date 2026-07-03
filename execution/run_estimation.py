import os
import time
from collections import Counter
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from data_reader import read_dataset
from metrics import l1_error, l2_error, kl_divergence
from metrics.EMD import earth_movers_distance
from utils.tabulation import print_table
from utils.calculate_win_counts import calculate_win_counts_and_percentages
from utils.file_operations import write_dict_to_csv

from core.ldp_engine import run_ldp_clean, resolve_protocols
from core.postprocessor import postprocess, resolve_methods


METRIC_MAP = {
    "l1":  l1_error,
    "l2":  l2_error,
    "kl":  kl_divergence,
    "emd": earth_movers_distance,
}

METRIC_PLOT_MAP = {
    "l1":  (r"$\ell_1$ Distance ($\times 10^{-3}$)", 1e3),
    "l2":  (r"$\ell_2$ Distance", 1.0),
    "kl":  ("KL Divergence", 1.0),
    "emd": ("EMD", 1.0),
}

_LABEL_MAP = {
    "base_pos": "Base-Pos",
    "norm":     "Norm",
    "norm_cut": "Norm-Cut",
    "norm_mul": "Norm-Mul",
    "norm_sub": "Norm-Sub",
    "power":    "Power",
    "power_ns": "Power-NS",
}

_COLORS = {
    "base_pos": "#9467bd",
    "norm":     "#1f77b4",
    "norm_cut": "#2ca02c",
    "norm_mul": "#d62728",
    "norm_sub": "#ff7f0e",
    "power":    "#8c564b",
    "power_ns": "#e377c2",
}

_MARKERS = {
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
]


def run_estimation(args, cfg):

    start = time.time()

    protocols      = resolve_protocols(cfg["experiment"]["protocols"])
    methods        = resolve_methods(cfg["experiment"]["methods"])
    epsilon_values = cfg["experiment"]["epsilon"]
    repeats        = cfg["experiment"]["repeats"]
    thread_number  = cfg["threading"]["num_threads"]
    utility_metric = cfg["experiment"]["utility_metric"].lower()
    dataset_path   = cfg["dataset"]["path"]
    save_csv       = cfg["output"].get("save_csv", False)
    csv_dir        = cfg["output"].get("csv_dir", "results")
    save_plots     = cfg["output"].get("save_plots", False)
    plot_dir       = cfg["output"].get("plot_dir", "plots")

    error_func = METRIC_MAP.get(utility_metric)
    if error_func is None:
        raise ValueError(
            f"Unknown utility metric '{utility_metric}'. "
            f"Available: {sorted(METRIC_MAP.keys())}"
        )

    full_path    = os.path.join(os.getcwd(), dataset_path)
    dataset_name = os.path.splitext(os.path.basename(full_path))[0]

    original_data, domain_size = read_dataset.read_data(full_path)
    n = len(original_data)

    counts    = Counter(original_data)
    frequency = [counts.get(i, 0) / n for i in range(domain_size)]

    _print_settings(
        epsilon_values, protocols, methods, repeats,
        thread_number, dataset_name, n, domain_size, utility_metric,
        save_plots,
    )


    error_dictionary  = {}
    error_without_pp  = {(p, eps): [] for p in protocols for eps in epsilon_values}

    for p in protocols:
        print(f"\n{'=' * 60}")
        print(f"Protocol: {p.upper()}")
        print(f"{'=' * 60}")

        for i in range(repeats):
            print(f"  Repeat {i + 1}/{repeats}")

            ldp_results = run_ldp_clean(
                protocol=p,
                epsilon=epsilon_values,
                thread_number=thread_number,
                input_data=deepcopy(original_data),
                domain_size=domain_size,
            )

            for eps in epsilon_values:
                estimated_counts = ldp_results[eps]["counts"]
                estimated_normal = ldp_results[eps]["normalized"]

                error_without_pp[(p, eps)].append(
                    error_func(estimated_normal, frequency)
                )

                for m in methods:
                    error, _ = postprocess(
                        method=m,
                        normalized_original=frequency,
                        estimated_counts=estimated_counts,
                        estimated_counts_normalized=estimated_normal,
                        original_data=original_data,
                        epsilon=eps,
                        protocol=p,
                        metric=error_func,
                    )
                    error_dictionary[(p, m, eps, i)] = error

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    for eps in epsilon_values:
        
        eps_errors = {
            (p, m, i): error_dictionary[(p, m, eps, i)]
            for p in protocols for m in methods for i in range(repeats)
        }
        avg_without_pp = {
            p: float(np.mean(error_without_pp[(p, eps)])) for p in protocols
        }
        std_without_pp = {
            p: round(float(np.std(error_without_pp[(p, eps)])), 6)
            for p in protocols
        }

        _, percentages, _ = calculate_win_counts_and_percentages(
            eps_errors, repeats
        )

        print(f"\nε = {eps}")
        print(f"{'-' * 60}")
        print_table(
            eps_errors,
            percentages,
            avg_without_pp,
            std_without_pp,
        )

        if save_csv:
            os.makedirs(csv_dir, exist_ok=True)
            filename = os.path.join(
                csv_dir,
                f"{dataset_name}_estimation_{utility_metric}_eps{eps}.csv",
            )
            write_dict_to_csv(eps_errors, avg_without_pp, filename)
            print(f"\nResults saved to: {filename}")

    if save_plots:
        _plot_estimation(
            error_dictionary, error_without_pp, protocols, methods,
            epsilon_values, repeats, dataset_name, plot_dir, utility_metric,
        )

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Total execution time: {elapsed:.2f} seconds")
    print(f"{'=' * 60}")



def _plot_estimation(
    error_dictionary, error_without_pp, protocols, methods,
    epsilon_values, repeats, dataset_name, plot_dir, utility_metric,
):

    os.makedirs(plot_dir, exist_ok=True)
    positions = np.arange(len(epsilon_values))
    metric_label, scale = METRIC_PLOT_MAP.get(
        utility_metric, (utility_metric.upper(), 1.0)
    )

    for p in protocols:
        plt.figure(figsize=(8, 6))

        for m in methods:
            avgs = [
                scale * float(np.mean([
                    error_dictionary[(p, m, eps, i)] for i in range(repeats)
                ]))
                for eps in epsilon_values
            ]
            plt.plot(
                positions,
                avgs,
                label=_LABEL_MAP.get(m, m),
                color=_COLORS.get(m),
                marker=_MARKERS.get(m, "o"),
                linewidth=4,
                markersize=12,
            )

        no_pp_avgs = [
            scale * float(np.mean(error_without_pp[(p, eps)]))
            for eps in epsilon_values
        ]
        plt.plot(
            positions,
            no_pp_avgs,
            label="No PP",
            color="black",
            linestyle="--",
            linewidth=4,
        )

        plt.xlabel(r"$\varepsilon$", fontsize=30)
        plt.ylabel(metric_label, fontsize=28)

        plt.xticks(
            positions,
            [rf"${e}$" for e in epsilon_values],
            fontsize=24,
        )
        plt.yticks(fontsize=22)

        handles, labels = plt.gca().get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))
        ordered_handles = []
        ordered_labels  = []
        for m in _LEGEND_ORDER:
            lbl = _LABEL_MAP.get(m, m)
            if lbl in label_to_handle:
                ordered_handles.append(label_to_handle[lbl])
                ordered_labels.append(lbl)
        if "No PP" in label_to_handle:
            ordered_handles.append(label_to_handle["No PP"])
            ordered_labels.append("No PP")

        plt.legend(
            ordered_handles, ordered_labels,
            fontsize=21, loc="best", ncol=2,
            columnspacing=1.5, handletextpad=0.6,
        )
        plt.grid(True, alpha=0.9, linestyle="-")
        plt.tight_layout()

        filename = os.path.join(
            plot_dir, f"{utility_metric}_{p}_{dataset_name}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"  Plot saved to: {filename}")
        plt.close()



def _print_settings(
    epsilon_values, protocols, methods, repeats,
    thread_number, dataset_name, n, domain_size, utility_metric,
    save_plots,
):
    print("=" * 60)
    print("FREQUENCY ESTIMATION")
    print("=" * 60)
    print(f"  Epsilon        : {epsilon_values}")
    print(f"  Protocols      : {protocols}")
    print(f"  Methods        : {methods}")
    print(f"  Repeats        : {repeats}")
    print(f"  Threads        : {thread_number}")
    print(f"  Dataset        : {dataset_name}")
    print(f"  Dataset size   : {n}")
    print(f"  Domain size    : {domain_size}")
    print(f"  Utility metric : {utility_metric.upper()}")
    print(f"  Save plots     : {save_plots}")
    print("=" * 60)
