import math

import numpy as np
import xxhash


def olh_g(epsilon):
    return int(round(math.exp(epsilon))) + 1


def ss_k(epsilon, domain_size):
    k = int(math.ceil(domain_size / (math.exp(epsilon) + 1)))
    return max(1, min(k, domain_size))


def _argmax_with_prior(log_likelihood, log_prior):
    if log_prior is not None:
        log_likelihood = log_likelihood + log_prior
    return int(np.argmax(log_likelihood))


def attack_grr(observations, epsilon, domain_size, log_prior=None):
    counts = np.zeros(domain_size)
    for y in observations:
        counts[y] += 1
    log_likelihood = epsilon * counts
    return _argmax_with_prior(log_likelihood, log_prior)


def attack_lh(observations, epsilon, domain_size, g, log_prior=None):
    match = np.zeros(domain_size)
    for xp, seed in observations:
        for v in range(domain_size):
            if xxhash.xxh32(str(v), seed=seed).intdigest() % g == xp:
                match[v] += 1
    log_likelihood = epsilon * match
    return _argmax_with_prior(log_likelihood, log_prior)


def attack_blh(observations, epsilon, domain_size, log_prior=None):
    return attack_lh(observations, epsilon, domain_size, g=2, log_prior=log_prior)


def attack_olh(observations, epsilon, domain_size, log_prior=None):
    return attack_lh(
        observations, epsilon, domain_size, g=olh_g(epsilon), log_prior=log_prior
    )


def attack_rappor(observations, epsilon, domain_size, log_prior=None):
    ones = np.zeros(domain_size)
    for bits in observations:
        ones += np.asarray(bits)
    log_likelihood = (epsilon / 2) * ones
    return _argmax_with_prior(log_likelihood, log_prior)


def attack_oue(observations, epsilon, domain_size, log_prior=None):
    ones = np.zeros(domain_size)
    for bits in observations:
        ones += np.asarray(bits)
    weight = math.log((math.exp(epsilon) + 1) / 2)
    log_likelihood = weight * ones
    return _argmax_with_prior(log_likelihood, log_prior)


def attack_ss(observations, epsilon, domain_size, log_prior=None):
    k = ss_k(epsilon, domain_size)
    e_eps = math.exp(epsilon)
    sigma = (k * e_eps) / (k * e_eps + domain_size - k)
    theta = ((k - 1) * (k * e_eps) + (domain_size - k) * k) / (
        (domain_size - 1) * (k * e_eps + domain_size - k)
    )
    weight = math.log(sigma / theta) - math.log((1.0 - sigma) / (1.0 - theta))

    counts = np.zeros(domain_size)
    for z in observations:
        for v in z:
            counts[v] += 1
    log_likelihood = weight * counts
    return _argmax_with_prior(log_likelihood, log_prior)


ATTACK_MAP = {
    "grr": attack_grr,
    "blh": attack_blh,
    "olh": attack_olh,
    "rappor": attack_rappor,
    "oue": attack_oue,
    "subset": attack_ss,
}


def get_attack(protocol):
    protocol = protocol.lower()
    if protocol not in ATTACK_MAP:
        raise ValueError(
            f"Unknown protocol '{protocol}'. Available: {sorted(ATTACK_MAP.keys())}"
        )
    return ATTACK_MAP[protocol]
