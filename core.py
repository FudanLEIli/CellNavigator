"""Generic numerical operations extracted and parameterized for public examples.

Inputs are arrays supplied by the caller. No data loading, application-specific
weights, biological identifiers, model fitting or file output occurs here.
"""
import math
from numbers import Integral
import numpy as np


def _matrix(value, name):
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or min(a.shape) == 0 or not np.isfinite(a).all():
        raise ValueError(f'{name} must be a nonempty finite 2D array')
    return a


def _pair(left, right):
    left, right = _matrix(left, 'left'), _matrix(right, 'right')
    if left.shape[1] != right.shape[1]:
        raise ValueError('Feature dimensions must match')
    return left, right


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def summarize(observations):
    """Return mean, population variance and count for an observations-by-features array."""
    x = _matrix(observations, 'observations')
    return x.mean(axis=0), x.var(axis=0, ddof=0), len(x)


def control_effects(means, variances, counts, *, control_index, prior_strength, scale_floor):
    """Construct raw and standardized control-relative mean/dispersion effects.

    All rows, including the control, are retained. The caller supplies a positive
    scale floor and nonnegative shrinkage strength; neither is selected here.
    The shrinkage rule uses population variances weighted by max(count - 1, 1).
    """
    means, variances = _pair(means, variances)
    counts = np.asarray(counts, dtype=float)
    if means.shape != variances.shape or np.any(variances < 0):
        raise ValueError('Means and nonnegative variances must have identical shapes')
    if counts.shape != (len(means),) or not np.isfinite(counts).all() or np.any(counts < 1) or np.any(counts != np.floor(counts)):
        raise ValueError('Counts must be positive integers, one per row')
    control_index = _integer(control_index, 'control_index')
    if control_index >= len(means):
        raise ValueError('Control index is out of bounds')
    if not np.isfinite(prior_strength) or prior_strength < 0:
        raise ValueError('prior_strength must be finite and nonnegative')
    if not np.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError('scale_floor must be finite and positive')
    weight = np.maximum(counts[:, None] - 1, 1)
    shrunk = (weight * variances + prior_strength * variances[control_index]) / (weight + prior_strength)
    mean = means - means[control_index]
    dispersion = np.sqrt(shrunk) - np.sqrt(shrunk[control_index])
    scale = np.maximum(np.sqrt(variances[control_index]), scale_floor)
    return {'mean': mean, 'dispersion': dispersion,
            'scaled_mean': mean / scale, 'scaled_dispersion': dispersion / scale}


def squared_distance(left, right):
    """Pairwise mean squared distances between rows; lower is closer."""
    left, right = _pair(left, right)
    distance = (np.square(left).sum(axis=1, keepdims=True)
                + np.square(right).sum(axis=1)[None, :] - 2 * left @ right.T) / left.shape[1]
    return np.maximum(distance, 0)


def cosine_similarity(left, right):
    """Pairwise row cosine similarity; a zero vector has similarity zero."""
    left, right = _pair(left, right)
    ln, rn = np.linalg.norm(left, axis=1, keepdims=True), np.linalg.norm(right, axis=1, keepdims=True)
    ln[ln == 0], rn[rn == 0] = 1, 1
    return (left / ln) @ (right / rn).T


def state_distance(query_mean, candidate_mean, query_dispersion, candidate_dispersion, *, dispersion_weight):
    """Combine component distances using an explicitly supplied nonnegative weight."""
    qm, cm = _pair(query_mean, candidate_mean)
    qd, cd = _pair(query_dispersion, candidate_dispersion)
    if qm.shape != qd.shape or cm.shape != cd.shape:
        raise ValueError('Mean and dispersion shapes must match')
    if not np.isfinite(dispersion_weight) or dispersion_weight < 0:
        raise ValueError('dispersion_weight must be finite and nonnegative')
    return squared_distance(qm, cm) + dispersion_weight * squared_distance(qd, cd)


def fuse_ranks(score_matrices, *, weights):
    """Fuse lower-is-better matrices using caller-supplied simplex weights.

    Ties use candidate column order. Scores and candidate order must be aligned.
    This function does not learn or select weights.
    """
    matrices = [_matrix(x, 'score matrix') for x in score_matrices]
    if not matrices or any(x.shape != matrices[0].shape for x in matrices):
        raise ValueError('Provide at least one score matrix, all of the same shape')
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(matrices),) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError('Weights must be nonnegative, finite and sum to one')
    ranks = [np.argsort(np.argsort(x, axis=1, kind='stable'), axis=1, kind='stable')
             / max(x.shape[1] - 1, 1) for x in matrices]
    return sum(w * r for w, r in zip(weights, ranks))


def hit_at_budget(scores, endpoint_distances, *, budget, relevant_count):
    """Return one hit indicator per query against a separately measured endpoint.

    Lower scores/distances are better. The caller must exclude ineligible/self
    candidates and keep endpoint measurements independent of ranking. Ties are
    resolved by candidate column order; budget is capped at library size.
    """
    scores, endpoint = _pair(scores, endpoint_distances)
    if scores.shape != endpoint.shape:
        raise ValueError('Scores and endpoint distances must have identical shapes')
    budget = _integer(budget, 'budget')
    relevant_count = _integer(relevant_count, 'relevant_count')
    n = scores.shape[1]
    if relevant_count > n:
        raise ValueError('Relevant count exceeds candidate count')
    selected = np.argsort(scores, axis=1, kind='stable')[:, :min(budget, n)]
    relevant = np.argsort(endpoint, axis=1, kind='stable')[:, :relevant_count]
    return np.array([np.isin(s, r).any() for s, r in zip(selected, relevant)])


def random_hit_probability(candidate_count, relevant_count, budget):
    """Exact hypergeometric probability of at least one hit without replacement."""
    n = _integer(candidate_count, 'candidate_count', 1)
    r = _integer(relevant_count, 'relevant_count')
    b = min(_integer(budget, 'budget'), n)
    if r > n:
        raise ValueError('Relevant count exceeds candidate count')
    if b == 0 or r == 0:
        return 0.0
    if b > n - r:
        return 1.0
    log_miss = math.fsum([math.lgamma(n-r+1), math.lgamma(n-b+1),
                          -math.lgamma(n-r-b+1), -math.lgamma(n+1)])
    return float(np.clip(-math.expm1(log_miss), 0, 1))
