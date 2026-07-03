import numpy as np
import xxhash
from sys import maxsize

_SEARCH_ITERATIONS = 1000


def poison_OLH(domain_size, epsilon, target_set=None):

    g = int(round(np.exp(epsilon))) + 1

    if target_set is not None and len(target_set) > 0:
        target_vals = list(target_set)
        use_target  = True
    else:
        target_vals = []
        use_target  = False

    entropy_list = []

    for _ in range(_SEARCH_ITERATIONS):

        rnd_seed = int(np.random.randint(0, maxsize, dtype=np.int64))

        if use_target:

            counts = np.zeros(g, dtype=int)
            for tv in target_vals:
                idx = xxhash.xxh32(str(tv), seed=rnd_seed).intdigest() % g
                counts[idx] += 1

            best_count = int(np.max(counts))
            best_index = int(np.argmax(counts))

            if best_count == len(target_vals):
                return best_index, rnd_seed

            entropy_list.append((best_count, rnd_seed, best_index))

        else:

            bucket_counts = np.zeros(g, dtype=int)
            for v in range(domain_size):
                idx = xxhash.xxh32(str(v), seed=rnd_seed).intdigest() % g
                bucket_counts[idx] += 1

            max_count = int(np.max(bucket_counts))
            max_index = int(np.argmax(bucket_counts))

            if max_count == domain_size:
                return max_index, rnd_seed

            entropy_list.append((max_count, rnd_seed, max_index))

    if not entropy_list:
        raise RuntimeError(
            "poison_OLH: entropy_list is empty after search. "
            "This should never happen — check _SEARCH_ITERATIONS."
        )

    best = max(entropy_list, key=lambda x: x[0])
    return best[2], best[1]   # (index, seed)


def poison_OUE(domain_size, epsilon, target_set=None):
   
    if target_set is not None and len(target_set) > 0:
        targets = list(target_set)
    else:
        targets = []

    r = len(targets)

    p = 0.5
    q = 1.0 / (np.exp(epsilon) + 1.0)

    # Expected number of 1s in a genuine OUE report: p + (d-1)q.
    expected_ones = int(np.floor(p + (domain_size - 1) * q))

    bit_vector = [0] * domain_size
    target_lookup = set(targets)
    for t in targets:
        bit_vector[t] = 1


    non_targets = [v for v in range(domain_size) if v not in target_lookup]
    l = max(0, min(expected_ones - r, len(non_targets)))

    if l > 0:
        chosen = np.random.choice(non_targets, size=l, replace=False)
        for idx in chosen:
            bit_vector[int(idx)] = 1

    return bit_vector


def poison_RAPPOR(domain_size, epsilon, target_set=None):
    
    if target_set is not None and len(target_set) > 0:
        targets = list(target_set)
    else:
        targets = []

    r = len(targets)

    p = np.exp(epsilon / 2.0) / (np.exp(epsilon / 2.0) + 1.0)
    q = 1.0 / (np.exp(epsilon / 2.0) + 1.0)

    expected_ones = int(np.floor(p + (domain_size - 1) * q))

    bit_vector = [0] * domain_size
    target_lookup = set(targets)
    for t in targets:
        bit_vector[t] = 1

    non_targets = [v for v in range(domain_size) if v not in target_lookup]
    l = max(0, min(expected_ones - r, len(non_targets)))

    if l > 0:
        chosen = np.random.choice(non_targets, size=l, replace=False)
        for idx in chosen:
            bit_vector[int(idx)] = 1

    return bit_vector


POISON_MAP = {
    "olh": poison_OLH,
    "oue": poison_OUE,
    "rappor": poison_RAPPOR,
}


def resolve_poison_func(protocol):
    if protocol not in POISON_MAP:
        raise ValueError(
            f"MGA poisoning is not implemented for protocol '{protocol}'. "
            f"Available: {sorted(POISON_MAP)}."
        )
    return POISON_MAP[protocol]
