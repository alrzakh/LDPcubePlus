import os
import time
from collections import Counter
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from data_reader import read_dataset
from metrics.gain import gain_reduction
from malicious_population.mga_poison import resolve_poison_func

from core.ldp_engine import run_ldp_poisoned, resolve_protocols
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



def run_poisoning(args, cfg):
    
    start = time.time()

    protocols        = resolve_protocols(cfg["experiment"]["protocols"])
    methods          = resolve_methods(cfg["experiment"]["methods"])
    epsilon_values   = cfg["experiment"]["epsilon"]
    repeats          = cfg["experiment"]["repeats"]
    thread_number    = cfg["threading"]["num_threads"]
    num_malicious    = cfg["adversarial"]["num_malicious"]
    target_selection = cfg["adversarial"]["target_selection"]
    target_k         = cfg["adversarial"]["target_k"]
    manual_targets   = cfg["adversarial"].get("manual_targets", [])
    save_plots       = cfg["output"].get("save_plots", False)
    plot_dir         = cfg["output"].get("plot_dir", "plots")
    dataset_path     = cfg["dataset"]["path"]

    full_path    = os.path.join(os.getcwd(), dataset_path)
    dataset_name = os.path.splitext(os.path.basename(full_path))[0]

    original_data, domain_size = read_dataset.read_data(full_path)
    n      = len(original_data)
    counts = Counter(original_data)
    frequency = [counts.get(i, 0) / n for i in range(domain_size)]

    targets = _resolve_targets(
        target_selection, target_k, manual_targets,
        domain_size, counts, original_data,
    )

    _print_settings(
        epsilon_values, protocols, methods, repeats,
        thread_number, dataset_name, n, domain_size,
        num_malicious, target_selection, targets,
    )

    for p in protocols:
        gain_vs_epsilon = {m: [] for m in methods}

        poison_func = resolve_poison_func(p)

        print(f"\n{'=' * 60}")
        print(f"Protocol: {p.upper()}")
        print(f"{'=' * 60}")

        for epsilon in epsilon_values:
            print(f"\n  --- ε = {epsilon} ---")

            gains_per_repeat = {m: [] for m in methods}

            for i in range(repeats):
                print(f"    Repeat {i + 1}/{repeats}")

                result = run_ldp_poisoned(
                    protocol=p,
                    epsilon=epsilon,
                    thread_number=thread_number,
                    input_data=deepcopy(original_data),
                    domain_size=domain_size,
                    number_of_malicious=num_malicious,
                    targets=targets,
                    poison_func=poison_func,
                )

                poisoned_counts = result["poisoned_counts"]
                poisoned_normal = result["poisoned_normalized"]

                for m in methods:
                    post_freq = postprocess(
                        method=m,
                        normalized_original=frequency,
                        estimated_counts=poisoned_counts,
                        estimated_counts_normalized=poisoned_normal,
                        original_data=original_data,
                        epsilon=epsilon,
                        protocol=p,
                        metric=None,
                    )

                    gain = gain_reduction(targets, poisoned_normal, post_freq)
                    gains_per_repeat[m].append(gain)

            for m in methods:
                avg_gain = sum(gains_per_repeat[m]) / repeats
                gain_vs_epsilon[m].append(avg_gain)
                print(f"    Avg GR for {m}: {avg_gain:.6f}")

        _print_summary_table(p, epsilon_values, methods, gain_vs_epsilon)

        if save_plots:
            target_tag = _target_tag(target_selection, target_k)
            _plot_gain_vs_epsilon(
                epsilon_values, gain_vs_epsilon, methods,
                p, dataset_name, plot_dir, target_tag,
            )

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Total execution time: {elapsed:.2f} seconds")
    print(f"{'=' * 60}")



def _resolve_targets(selection, k, manual, domain_size, counts, original_data):
   
    sorted_by_freq = sorted(range(domain_size), key=lambda i: counts.get(i, 0))

    if selection == "bottom_k":
        targets = sorted_by_freq[:k]
        print(f"Targets: bottom-{k} least frequent items → {targets}")

    elif selection == "top_k":
        targets = sorted_by_freq[-k:]
        print(f"Targets: top-{k} most frequent items → {targets}")

    elif selection == "all":
        targets = list(range(domain_size))
        print(f"Targets: all {domain_size} domain items (untargeted)")

    elif selection == "manual":
        if not manual:
            raise ValueError(
                "target_selection is 'manual' but 'manual_targets' is empty. "
                "Provide a list of target domain values in mga.yaml."
            )
        targets = list(manual)
        print(f"Targets: manual list → {targets}")

    else:
        raise ValueError(
            f"Unknown target_selection '{selection}'. "
            "Choose from: bottom_k, top_k, all, manual."
        )

    return targets



def _print_settings(
    epsilon_values, protocols, methods, repeats,
    thread_number, dataset_name, n, domain_size,
    num_malicious, target_selection, targets,
):
    print("=" * 60)
    print("MGA POISONING ATTACK")
    print("=" * 60)
    print(f"  Epsilon values   : {epsilon_values}")
    print(f"  Protocols        : {protocols}")
    print(f"  Methods          : {methods}")
    print(f"  Repeats          : {repeats}")
    print(f"  Threads          : {thread_number}")
    print(f"  Dataset          : {dataset_name}")
    print(f"  Dataset size     : {n}")
    print(f"  Domain size      : {domain_size}")
    print(f"  Malicious users  : {num_malicious}")
    print(f"  Target strategy  : {target_selection}")
    print(f"  Target count     : {len(targets)}")
    print("=" * 60)


def _print_summary_table(protocol, epsilon_values, methods, gain_vs_epsilon):
    print(f"\n{'=' * 60}")
    print(f"Summary — Protocol: {protocol.upper()}")
    print(f"{'=' * 60}")
    header = f"{'Method':<15} | " + " | ".join(
        [f"ε={eps:<5}" for eps in epsilon_values]
    )
    print(header)
    print("-" * len(header))
    for m in methods:
        gains_str = " | ".join([f"{g:.4f}" for g in gain_vs_epsilon[m]])
        print(f"{m:<15} | {gains_str}")


def _target_tag(selection, k):
    if selection in ("bottom_k", "top_k"):
        return f"{selection[:-2]}_{k}"   # 'bottom_k' -> 'bottom_10'
    return selection                     # 'all' / 'manual'


def _plot_gain_vs_epsilon(
    epsilon_values, gain_results, methods,
    protocol, dataset_name, output_dir, target_tag,
):

    os.makedirs(output_dir, exist_ok=True)
    positions = np.arange(len(epsilon_values))

    plt.figure(figsize=(8, 6))

    for method in methods:
        if method in gain_results:
            plt.plot(
                positions,
                gain_results[method],
                label=_LABEL_MAP.get(method, method),
                color=_COLORS.get(method),
                marker=_MARKERS.get(method, "o"),
                linewidth=4,
                markersize=12,
            )

    plt.xlabel(r"$\varepsilon$", fontsize=30)
    plt.ylabel("Gain Reduction", fontsize=28)


    plt.xticks(
        positions,
        [rf"${e}$" for e in epsilon_values],
        fontsize=24,
    )
    plt.yticks(fontsize=22)

    handles, labels = plt.gca().get_legend_handles_labels()
    method_to_handle = dict(zip(methods, handles))
    ordered_handles = []
    ordered_labels  = []
    for m in _LEGEND_ORDER:
        if m in method_to_handle:
            ordered_handles.append(method_to_handle[m])
            ordered_labels.append(_LABEL_MAP.get(m, m))

    plt.legend(
        ordered_handles, ordered_labels,
        fontsize=21, loc="best", ncol=2,
        columnspacing=1.5, handletextpad=0.6,
    )
    plt.grid(True, alpha=0.9, linestyle="-")
    plt.tight_layout()

    filename = os.path.join(
        output_dir, f"GR_{protocol}_{dataset_name}_{target_tag}.png"
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"  Plot saved to: {filename}")
    plt.close()
