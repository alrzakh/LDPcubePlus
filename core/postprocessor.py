from post_processing import norm, norm_mul, norm_cut, norm_sub, base_pos, power
from protocols.variance import calculate_variances

ALL_METHODS = ["base_pos", "norm", "norm_cut", "norm_mul", "norm_sub", "power", "power_ns"]

_METHOD_MAP = {
    "base_pos": base_pos,
    "norm":     norm,
    "norm_cut": norm_cut,
    "norm_mul": norm_mul,
    "norm_sub": norm_sub,
    "power":    power,
    "power_ns": power,
}


def postprocess(
    method,
    normalized_original,
    estimated_counts,
    estimated_counts_normalized,
    original_data,
    epsilon,
    protocol,
    metric=None,
):

    func = _METHOD_MAP.get(method)
    if func is None:
        raise ValueError(
            f"Unknown post-processing method '{method}'. "
            f"Available: {sorted(_METHOD_MAP.keys())}"
        )

    dom_size = len(set(original_data))
    pop_size = len(original_data)

    if method in ("power", "power_ns"):
        variance  = calculate_variances(protocol, epsilon, dom_size, pop_size)
        post_data = func(estimated_counts, pop_size, variance)
        if method == "power_ns":
            post_data = norm_sub(post_data, normalized_original)
    else:
        post_data = func(estimated_counts_normalized, normalized_original)

    if metric is not None:
        error = metric(post_data, normalized_original)
        return error, post_data

    return post_data


def resolve_methods(requested):

    if "all" in requested:
        if len(requested) > 1:
            print("Warning: 'all' overrides specific methods. Using all methods.")
        return ALL_METHODS
    return requested
