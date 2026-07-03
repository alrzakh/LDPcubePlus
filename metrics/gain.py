import numpy as np

def gain_reduction(targets, f_attacked, f_recovery, use_abs=False):

    f_attacked  = np.asarray(f_attacked, dtype=float)
    f_recovery  = np.asarray(f_recovery, dtype=float)

    idx = np.asarray(list(targets), dtype=int)

    diff = f_attacked[idx] - f_recovery[idx]
    if use_abs:
        diff = np.abs(diff)

    return float(diff.sum())
