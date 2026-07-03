import numpy as np


def norm(est_dist ,org_dist):
    estimates = np.copy(est_dist)
    total = estimates.sum()
    d = len(estimates)
    diff = (1 - total) / d
    return estimates + diff
