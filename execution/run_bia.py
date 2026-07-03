import os
import time
from collections import Counter
from copy import deepcopy
from concurrent import futures

import numpy as np

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


def _binary_search_worker(task):

    (
        protocol, method, real_frequencies, original_data,
        thread_number, domain_size, difference_observed,
        epsilon_min, epsilon_max, tolerance,
    ) = task

    n = len(original_data)


    synthetic_set = []
    for i in range(domain_size):
        count = int(real_frequencies[i] * n)
        synthetic_set.extend([i] * count)
    np.random.shuffle(synthetic_set)

    eps_lo = epsilon_min
    eps_hi = epsilon_max

    while eps_hi - eps_lo > tolerance:
        eps_mid = (eps_hi + eps_lo) / 2

        result = run_ldp_clean(
            protocol=protocol,
            epsilon=eps_mid,
            thread_number=thread_number,
            input_data=deepcopy(synthetic_set),
            domain_size=domain_size,
        )

        post_freq = postprocess(
            method=method,
            normalized_original=real_frequencies,
            estimated_counts=result["counts"],
            estimated_counts_normalized=result["normalized"],
            original_data=original_data,
            epsilon=eps_mid,
            protocol=protocol,
            metric=None,
        )

        difference_simulated = l1_error(post_freq, real_frequencies)


        if difference_simulated < difference_observed:
            eps_lo = eps_mid
        else:
            eps_hi = eps_mid

    return protocol, method, eps_lo


def run_bia(args, cfg):

    start = time.time()

    protocols      = resolve_protocols(cfg["experiment"]["protocols"])
    methods        = resolve_methods(cfg["experiment"]["methods"])
    epsilon_list   = cfg["experiment"]["epsilon"]
    epsilon        = epsilon_list[0]           # BIA uses a single true epsilon
    repeats        = cfg["experiment"]["repeats"]
    thread_number  = cfg["threading"]["num_threads"]
    utility_metric = cfg["experiment"]["utility_metric"].lower()
    dataset_path   = cfg["dataset"]["path"]
    save_csv       = cfg["output"].get("save_csv", False)
    csv_dir        = cfg["output"].get("csv_dir", "results")

    bs_cfg         = cfg.get("binary_search", {})
    run_bs         = bs_cfg.get("run", False)
    bs_runs        = bs_cfg.get("runs", 1)
    bs_eps_min     = bs_cfg.get("epsilon_min", 0.1)
    bs_eps_max     = bs_cfg.get("epsilon_max", 5.0)
    bs_tolerance   = bs_cfg.get("tolerance", 0.001)

    error_func = METRIC_MAP.get(utility_metric)
    if error_func is None:
        raise ValueError(
            f"Unknown utility metric '{utility_metric}'. "
            f"Available: {sorted(METRIC_MAP.keys())}"
        )

    full_path    = os.path.join(os.getcwd(), dataset_path)
    dataset_name = os.path.splitext(os.path.basename(full_path))[0]

    original_data, domain_size = read_dataset.read_data(full_path)
    n      = len(original_data)
    counts = Counter(original_data)
    frequency = [counts.get(i, 0) / n for i in range(domain_size)]

    _print_settings(
        epsilon, protocols, methods, repeats,
        thread_number, dataset_name, n, domain_size,
        utility_metric, run_bs,
    )


    error_dictionary = {}
    error_without_pp = {p: [] for p in protocols}


    estimated_store = {p: [] for p in protocols}

    for i in range(repeats):
        for p in protocols:
            print(f"  Protocol {p.upper()}, repeat {i + 1}/{repeats}")

            result = run_ldp_clean(
                protocol=p,
                epsilon=epsilon,
                thread_number=thread_number,
                input_data=deepcopy(original_data),
                domain_size=domain_size,
            )
            estimated_store[p].append(result)

            error_without_pp[p].append(
                error_func(result["normalized"], frequency)
            )

            for m in methods:
                error, _ = postprocess(
                    method=m,
                    normalized_original=frequency,
                    estimated_counts=result["counts"],
                    estimated_counts_normalized=result["normalized"],
                    original_data=original_data,
                    epsilon=epsilon,
                    protocol=p,
                    metric=error_func,
                )
                error_dictionary[(p, m, i)] = error

    print(f"\n  Phase 1 time: {time.time() - start:.2f}s")

    avg_error_without_pp = {
        p: float(np.mean(vals)) for p, vals in error_without_pp.items()
    }
    std_dev_without_pp = {
        p: round(float(np.std(vals)), 6) for p, vals in error_without_pp.items()
    }

    _, percentages, _ = calculate_win_counts_and_percentages(
        error_dictionary, repeats
    )

    print(f"\n{'=' * 60}")
    print("PHASE 1 RESULTS — Frequency Estimation")
    print(f"{'=' * 60}\n")
    print_table(
        error_dictionary,
        percentages,
        avg_error_without_pp,
        std_dev_without_pp,
    )

    if save_csv:
        os.makedirs(csv_dir, exist_ok=True)
        filename = os.path.join(
            csv_dir, f"{dataset_name}_bia_estimation_{utility_metric}.csv"
        )
        write_dict_to_csv(error_dictionary, avg_error_without_pp, filename)
        print(f"\n  Results saved to: {filename}")

    if run_bs:
        _run_binary_search(
            protocols=protocols,
            methods=methods,
            frequency=frequency,
            original_data=original_data,
            thread_number=thread_number,
            domain_size=domain_size,
            estimated_store=estimated_store,
            true_epsilon=epsilon,
            bs_runs=bs_runs,
            eps_min=bs_eps_min,
            eps_max=bs_eps_max,
            tolerance=bs_tolerance,
        )

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Total execution time: {elapsed:.2f} seconds")
    print(f"{'=' * 60}")



def _run_binary_search(
    protocols, methods, frequency, original_data,
    thread_number, domain_size, estimated_store,
    true_epsilon, bs_runs, eps_min, eps_max, tolerance,
):

    bs_start = time.time()

    all_guesses = {(p, m): [] for p in protocols for m in methods}

    for run_idx in range(bs_runs):
        print(f"\n{'=' * 70}")
        print(f"Binary Search — Run {run_idx + 1}/{bs_runs}")
        print(f"{'=' * 70}")

        repeat_idx  = run_idx % len(estimated_store[protocols[0]])
        diff_dict   = {}

        for p in protocols:
            stored      = estimated_store[p][repeat_idx]
            est_counts  = stored["counts"]
            est_normal  = stored["normalized"]

            for m in methods:
                post_freq = postprocess(
                    method=m,
                    normalized_original=frequency,
                    estimated_counts=est_counts,
                    estimated_counts_normalized=est_normal,
                    original_data=original_data,
                    epsilon=true_epsilon,
                    protocol=p,
                    metric=None,
                )
                diff_dict[(p, m)] = l1_error(post_freq, frequency)

        tasks = [
            (
                p, m, frequency, original_data,
                thread_number, domain_size,
                diff_dict[(p, m)],
                eps_min, eps_max, tolerance,
            )
            for p in protocols
            for m in methods
        ]

        print(f"  Running {len(tasks)} parallel binary search tasks...")

        with futures.ProcessPoolExecutor() as executor:
            jobs = [
                executor.submit(_binary_search_worker, task)
                for task in tasks
            ]
            for job in futures.as_completed(jobs):
                p, m, guess = job.result()
                all_guesses[(p, m)].append(guess)
                l1_err = abs(guess - true_epsilon)
                print(
                    f"  ✓ {p.upper()}-{m}: "
                    f"ε_guess={guess:.4f}  "
                    f"ε_true={true_epsilon:.4f}  "
                    f"L1 err={l1_err:.4f}"
                )

    print(f"\n{'=' * 70}")
    print("PHASE 2 RESULTS — Epsilon Inference Summary")
    print(f"  Runs: {bs_runs}  |  True ε: {true_epsilon}")
    print(f"{'=' * 70}")

    for (p, m) in sorted(all_guesses.keys()):
        guesses    = all_guesses[(p, m)]
        mean_guess = float(np.mean(guesses))
        std_guess  = float(np.std(guesses)) if bs_runs > 1 else 0.0
        l1_err     = abs(mean_guess - true_epsilon)

        if bs_runs > 1:
            print(
                f"  {p.upper()}-{m:<12}: "
                f"ε_guess={mean_guess:.4f} ± {std_guess:.4f}  "
                f"L1 err={l1_err:.4f}"
            )
        else:
            print(
                f"  {p.upper()}-{m:<12}: "
                f"ε_guess={mean_guess:.4f}  "
                f"L1 err={l1_err:.4f}"
            )

    print(f"\n  Binary search time: {time.time() - bs_start:.2f}s")



def _print_settings(
    epsilon, protocols, methods, repeats,
    thread_number, dataset_name, n, domain_size,
    utility_metric, run_bs,
):
    print("=" * 60)
    print("BUDGET INFERENCE ATTACK (BIA)")
    print("=" * 60)
    print(f"  True epsilon     : {epsilon}")
    print(f"  Protocols        : {protocols}")
    print(f"  Methods          : {methods}")
    print(f"  Repeats          : {repeats}")
    print(f"  Threads          : {thread_number}")
    print(f"  Dataset          : {dataset_name}")
    print(f"  Dataset size     : {n}")
    print(f"  Domain size      : {domain_size}")
    print(f"  Utility metric   : {utility_metric.upper()}")
    print(f"  Binary search    : {'enabled' if run_bs else 'disabled'}")
    print("=" * 60)
