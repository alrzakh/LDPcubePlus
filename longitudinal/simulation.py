import random
from collections import Counter

import numpy as np

from core.ldp_engine import PROTOCOL_MAP
from longitudinal.attacks import get_attack, olh_g, ss_k


def simulate_observations(v_u, n, consistency, perturb_func, epsilon, domain_size):
    """
    Generate n perturbed observations for a single user.

    Returns
    -------
    observations : list      one perturbed report per timestamp
    label        : int       ground-truth value to score against (mode of the
                             realized true-value stream)
    """
    observations = []
    true_stream = []

    for _ in range(n):
        if consistency >= 1.0 or random.random() < consistency:
            tv = v_u
        else:
            tv = random.randint(0, domain_size - 1)
        true_stream.append(tv)
        observations.append(perturb_func(tv, epsilon, domain_size))

    if consistency >= 1.0:
        label = v_u
    else:

        counts = Counter(true_stream)
        best = max(counts.values())
        label = v_u if counts.get(v_u, 0) == best else counts.most_common(1)[0][0]

    return observations, label


def attack_user_chunk(task):
    
    (
        user_values, protocol, epsilon, n, consistency,
        domain_size, log_prior, seed,
    ) = task

    np.random.seed(seed)
    random.seed(seed)

    protocol = protocol.lower()
    perturb_func = PROTOCOL_MAP[protocol].perturb
    attack_func = get_attack(protocol)

    labels = []
    preds = []

    for v_u in user_values:
        observations, label = simulate_observations(
            v_u, n, consistency, perturb_func, epsilon, domain_size
        )
        pred = attack_func(observations, epsilon, domain_size, log_prior=log_prior)

        labels.append(label)
        preds.append(pred)

    return labels, preds


def estimate_population_frequencies(
    user_values, protocol, epsilon, domain_size, seed=None,
):
   
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    protocol = protocol.lower()
    module = PROTOCOL_MAP[protocol]

    reports = [module.perturb(v_u, epsilon, domain_size) for v_u in user_values]
    counts = module.aggregate(reports, epsilon, domain_size)
    return np.asarray(counts, dtype=float) / len(reports)


def protocol_param_str(protocol, epsilon, domain_size):
    protocol = protocol.lower()
    if protocol == "olh":
        return f"g={olh_g(epsilon)}"
    if protocol == "blh":
        return "g=2"
    if protocol == "subset":
        return f"k={ss_k(epsilon, domain_size)}"
    return ""
