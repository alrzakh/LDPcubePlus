import math


def asr(true_values, predicted_values):
    total = len(true_values)
    if total == 0:
        return 0.0
    correct = sum(1 for t, p in zip(true_values, predicted_values) if t == p)
    return correct / total


# baseline_adversaries
def random_asr(domain_size):
    return 1.0 / domain_size


def rr_bound_asr(epsilon, domain_size):
    e_eps = math.exp(epsilon)
    return e_eps / (e_eps + domain_size - 1)
