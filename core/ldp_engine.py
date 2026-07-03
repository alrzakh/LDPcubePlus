import random
from concurrent import futures

import numpy as np

from protocols import grr, rappor, olh, oue, blh, subset

PROTOCOL_MAP = {
    "grr":    grr,
    "rappor": rappor,
    "olh":    olh,
    "blh":    blh,
    "subset": subset,
    "oue":    oue,
}


def _clean_chunk(chunk, epsilon, perturb_func, aggregate_func, domain_size, seed):

    np.random.seed(seed)
    random.seed(seed)

    perturbed = [perturb_func(v, epsilon, domain_size) for v in chunk]
    return aggregate_func(perturbed, epsilon, domain_size)


def _poisoned_chunk(
    chunk,
    epsilon,
    perturb_func,
    aggregate_func,
    poison_func,
    domain_size,
    seed,
    num_malicious_this_thread,
    targets,
):

    np.random.seed(seed)
    random.seed(seed)

    perturbed = [perturb_func(v, epsilon, domain_size) for v in chunk]
    clean_counts = aggregate_func(perturbed, epsilon, domain_size)

    malicious_reports = [
        poison_func(domain_size, epsilon, targets)
        for _ in range(num_malicious_this_thread)
    ]

    poisoned_counts = aggregate_func(perturbed + malicious_reports, epsilon, domain_size)

    return clean_counts, poisoned_counts


def run_ldp_clean(protocol, epsilon, thread_number, input_data, domain_size):

    ldp_method   = _get_protocol(protocol)
    data_chunks  = np.array_split(input_data, thread_number)
    data_size    = len(input_data)
    perturb_func = ldp_method.perturb
    agg_func     = ldp_method.aggregate

    single_mode = isinstance(epsilon, (int, float))
    epsilon_list = [epsilon] if single_mode else list(epsilon)

    results_by_eps = {}

    for eps in epsilon_list:
        seeds = [random.randint(1, 100_000_000) for _ in range(thread_number)]

        chunk_results = []
        with futures.ProcessPoolExecutor() as executor:
            jobs = [
                executor.submit(
                    _clean_chunk,
                    chunk, eps, perturb_func, agg_func, domain_size, seed
                )
                for seed, chunk in zip(seeds, data_chunks)
            ]
            for job in futures.as_completed(jobs):
                chunk_results.append(job.result())

        counts     = [sum(vals) for vals in zip(*chunk_results)]
        normalized = [c / data_size for c in counts]

        results_by_eps[eps] = {"counts": counts, "normalized": normalized}

    if single_mode:
        return results_by_eps[epsilon_list[0]]
    return results_by_eps


def run_ldp_poisoned(
    protocol,
    epsilon,
    thread_number,
    input_data,
    domain_size,
    number_of_malicious,
    targets,
    poison_func,
):

    ldp_method   = _get_protocol(protocol)
    data_chunks  = np.array_split(input_data, thread_number)
    data_size    = len(input_data)
    perturb_func = ldp_method.perturb
    agg_func     = ldp_method.aggregate


    base_mal      = number_of_malicious // thread_number
    remainder     = number_of_malicious % thread_number
    mal_per_thread = [
        base_mal + (1 if i < remainder else 0)
        for i in range(thread_number)
    ]

    seeds = [random.randint(1, 100_000_000) for _ in range(thread_number)]

    clean_parts   = []
    poisoned_parts = []

    with futures.ProcessPoolExecutor() as executor:
        jobs = [
            executor.submit(
                _poisoned_chunk,
                chunk, epsilon, perturb_func, agg_func, poison_func,
                domain_size, seed, n_mal, targets,
            )
            for chunk, seed, n_mal in zip(data_chunks, seeds, mal_per_thread)
        ]
        for job in futures.as_completed(jobs):
            clean, poisoned = job.result()
            clean_parts.append(clean)
            poisoned_parts.append(poisoned)

    clean_counts   = [sum(vals) for vals in zip(*clean_parts)]
    poisoned_counts = [sum(vals) for vals in zip(*poisoned_parts)]

    total_size = data_size + number_of_malicious

    return {
        "clean_counts":       clean_counts,
        "clean_normalized":   [c / data_size  for c in clean_counts],
        "poisoned_counts":    poisoned_counts,
        "poisoned_normalized": [c / total_size for c in poisoned_counts],
    }


def _get_protocol(name):
    if name not in PROTOCOL_MAP:
        raise ValueError(
            f"Unknown protocol '{name}'. "
            f"Available: {sorted(PROTOCOL_MAP.keys())}"
        )
    return PROTOCOL_MAP[name]


def resolve_protocols(requested):

    all_protocols = list(PROTOCOL_MAP.keys())
    if "all" in requested:
        if len(requested) > 1:
            print("Warning: 'all' overrides specific protocols. Using all protocols.")
        return all_protocols
    return requested
