import numpy as np

def calculate_IGR(target_set,
                  f_base,        # list/array
                  f_before,      # list/array
                  f_recovery):   # list/array
    
    f_base     = np.asarray(f_base, dtype=float)
    f_before   = np.asarray(f_before, dtype=float)
    f_recovery = np.asarray(f_recovery, dtype=float)
    
    idx = np.asarray(list(target_set), dtype=int)
    
    max_len = min(f_base.shape[0], f_before.shape[0], f_recovery.shape[0])
    idx = idx[(idx >= 0) & (idx < max_len)]
    
    if idx.size == 0:
        return 0.0
    
    r = len(idx)
    
    if r == 0:
        return 0.0
    
    numerator = np.sum(f_recovery[idx] - f_before[idx])
    
    denominator = np.sum(f_base[idx] - f_before[idx]) * r
    
    if denominator == 0:
        return 0.0
    
    IGR = numerator / denominator
    
    return float(IGR)