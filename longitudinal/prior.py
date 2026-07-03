import numpy as np

from post_processing import base_pos, norm, norm_cut, norm_sub, norm_mul, power
from protocols.variance import calculate_variances


RAW_TOKENS = {"no_pp", "none", "raw"}
PP_METHODS = {"base_pos", "norm", "norm_cut", "norm_sub", "norm_mul", "power", "power_ns"}
ALL_PRIOR_TOKENS = ["uniform", "no_pp"] + sorted(PP_METHODS)


def _post_process(fhat, method, protocol, epsilon, domain_size, pop_size):
    
    target = np.ones(domain_size) / domain_size

    if method == "base_pos":
        return base_pos(fhat)
    if method == "norm":
        return norm(fhat, target)
    if method == "norm_cut":
        return norm_cut(fhat, target)
    if method == "norm_sub":
        return norm_sub(fhat, target)
    if method == "norm_mul":
        return norm_mul(fhat, target)
    if method in ("power", "power_ns"):
        var = calculate_variances(
            protocol.lower(), epsilon, domain_size, pop_size
        )
        counts = np.asarray(fhat) * pop_size          # Power expects count-scale input
        post = power(counts, pop_size, var)
        if method == "power_ns":
            post = norm_sub(post, target)
        return post

    raise ValueError(
        f"Unknown post-processing method '{method}'. "
        f"Available: {sorted(PP_METHODS)}"
    )


def _to_distribution(vec, domain_size):
    vec = np.clip(np.asarray(vec, dtype=float), 0.0, None)
    total = vec.sum()
    if total <= 0:
        return np.ones(domain_size) / domain_size
    return vec / total


def build_prior(fhat, method, protocol, epsilon, domain_size, pop_size):
   
    method = method.lower()

    if method == "uniform":
        return np.ones(domain_size) / domain_size

    if method in RAW_TOKENS:
        return _to_distribution(fhat, domain_size)

    post = _post_process(fhat, method, protocol, epsilon, domain_size, pop_size)
    return _to_distribution(post, domain_size)


def build_log_prior(fhat, method, protocol, epsilon, domain_size, pop_size, eps_floor=1e-12):
   
    if method.lower() == "uniform":
        return None
    prior = build_prior(fhat, method, protocol, epsilon, domain_size, pop_size)
    return np.log(prior + eps_floor)
