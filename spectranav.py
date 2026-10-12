"""Lightweight graph-spectral utilities for the SpectraNav MVP.

The module deliberately contains no access to confirmation-context data.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
from scipy.special import softmax
from scipy.special import logsumexp
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class SpectralBasis:
    low_vectors: np.ndarray
    high_vectors: np.ndarray
    low_values: np.ndarray
    high_values: np.ndarray
    n_features: int


def build_effect_graph(
    candidate_signatures: np.ndarray,
    *,
    embedding_dim: int = 64,
    neighbours: int = 12,
    seed: int = 20260724,
) -> sparse.csr_matrix:
    """Build a unified sparse gene graph from candidate training signatures.

    Rows are candidate perturbations and columns are aligned genes. A randomized
    SVD creates gene embeddings, followed by a mutualized cosine kNN graph.
    """
    x = np.asarray(candidate_signatures, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(np.square(x), axis=0))
    scale = np.maximum(scale, np.quantile(scale[scale > 0], 0.10))
    x = x / scale
    rank = min(embedding_dim, min(x.shape) - 1)
    svd = TruncatedSVD(n_components=rank, random_state=seed, n_iter=7)
    svd.fit(x)
    gene_embedding = svd.components_.T * svd.singular_values_[None, :]
    gene_embedding /= np.maximum(
        np.linalg.norm(gene_embedding, axis=1, keepdims=True), 1e-12
    )
    k = min(neighbours + 1, gene_embedding.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
    nn.fit(gene_embedding)
    distances, indices = nn.kneighbors(gene_embedding)
    rows = np.repeat(np.arange(gene_embedding.shape[0]), k - 1)
    cols = indices[:, 1:].reshape(-1)
    weights = np.clip(1.0 - distances[:, 1:].reshape(-1), 0.0, 1.0)
    graph = sparse.coo_matrix(
        (weights, (rows, cols)),
        shape=(gene_embedding.shape[0], gene_embedding.shape[0]),
    ).tocsr()
    graph = graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def normalized_laplacian(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degrees)
    positive = degrees > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    d = sparse.diags(inv_sqrt)
    identity = sparse.eye(adjacency.shape[0], format="csr")
    return identity - d @ adjacency @ d


def spectral_basis(
    adjacency: sparse.csr_matrix,
    *,
    band_rank: int = 64,
    seed: int = 20260724,
) -> SpectralBasis:
    """Compute explicit low- and high-frequency graph Fourier bases."""
    laplacian = normalized_laplacian(adjacency)
    rank = min(band_rank, adjacency.shape[0] // 3)
    low_values, low_vectors = eigsh(
        laplacian, k=rank + 1, which="SM", tol=1e-5,
        v0=np.random.default_rng(seed).normal(size=adjacency.shape[0])
    )
    # Drop the approximately constant first eigenvector.
    low_values, low_vectors = low_values[1:], low_vectors[:, 1:]
    high_values, high_vectors = eigsh(
        laplacian, k=rank, which="LA", tol=1e-5,
        v0=np.random.default_rng(seed + 1).normal(size=adjacency.shape[0])
    )
    return SpectralBasis(
        low_vectors=low_vectors.astype(np.float32),
        high_vectors=high_vectors.astype(np.float32),
        low_values=low_values.astype(np.float32),
        high_values=high_values.astype(np.float32),
        n_features=adjacency.shape[0],
    )


def degree_preserving_random_graph(
    adjacency: sparse.csr_matrix,
    *,
    seed: int = 20260724,
    swaps_per_edge: int = 4,
) -> sparse.csr_matrix:
    """Randomize topology with double-edge swaps while retaining node degrees."""
    upper = sparse.triu(adjacency, k=1).tocoo()
    graph = nx.Graph()
    graph.add_nodes_from(range(adjacency.shape[0]))
    graph.add_edges_from(zip(upper.row.tolist(), upper.col.tolist()))
    swaps = swaps_per_edge * graph.number_of_edges()
    nx.double_edge_swap(
        graph,
        nswap=swaps,
        max_tries=max(swaps * 20, 1000),
        seed=seed,
    )
    rows, cols = zip(*graph.edges())
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    values = np.ones(len(rows), dtype=np.float32)
    result = sparse.coo_matrix(
        (
            np.r_[values, values],
            (np.r_[rows, cols], np.r_[cols, rows]),
        ),
        shape=adjacency.shape,
    ).tocsr()
    return result


def project_bands(x: np.ndarray, basis: SpectralBasis) -> dict[str, np.ndarray]:
    """Return low/high Fourier coefficients and residual-band squared norms."""
    x = np.asarray(x, dtype=np.float32)
    low = x @ basis.low_vectors
    high = x @ basis.high_vectors
    total_norm = np.sum(np.square(x), axis=1)
    residual_norm = np.maximum(
        total_norm - np.sum(np.square(low), axis=1)
        - np.sum(np.square(high), axis=1),
        0.0,
    )
    return {"low": low, "high": high, "mid_norm": residual_norm}


def pairwise_squared(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(np.square(a), axis=1)[:, None]
    bb = np.sum(np.square(b), axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * a @ b.T, 0.0)


def band_distances(
    query: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    n_features: int,
) -> np.ndarray:
    """Return normalized low/mid/high squared distances."""
    low = pairwise_squared(query["low"], candidate["low"])
    high = pairwise_squared(query["high"], candidate["high"])
    total = (
        query["mid_norm"][:, None] + candidate["mid_norm"][None, :]
    )
    # The residual components are not explicitly stored. Their cross term is
    # approximated from total-space distances by the caller when available.
    return np.stack(
        [
            low / query["low"].shape[1],
            total / max(n_features - 2 * query["low"].shape[1], 1),
            high / query["high"].shape[1],
        ],
        axis=-1,
    )


def exact_band_distances(
    query_x: np.ndarray,
    candidate_x: np.ndarray,
    basis: SpectralBasis,
) -> np.ndarray:
    query = project_bands(query_x, basis)
    candidate = project_bands(candidate_x, basis)
    low = pairwise_squared(query["low"], candidate["low"])
    high = pairwise_squared(query["high"], candidate["high"])
    total = pairwise_squared(query_x, candidate_x)
    mid = np.maximum(total - low - high, 0.0)
    mid_dim = max(basis.n_features - 2 * query["low"].shape[1], 1)
    return np.stack(
        [
            low / query["low"].shape[1],
            mid / mid_dim,
            high / query["high"].shape[1],
        ],
        axis=-1,
    ).astype(np.float32)


def normalize_listwise_features(
    features: np.ndarray, reference_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    reference = features[reference_rows]
    scales = np.median(reference.reshape(-1, reference.shape[-1]), axis=0)
    scales = np.maximum(scales, 1e-8)
    return features / scales[None, None, :], scales


def listnet_weights(
    features: np.ndarray,
    actual_w2: np.ndarray,
    development_rows: np.ndarray,
    *,
    target_temperature: float = 0.30,
    prediction_temperature: float = 0.50,
    regularization: float = 0.02,
) -> tuple[np.ndarray, float]:
    """Fit non-negative spectral-band weights with a ListNet objective."""
    x, _ = normalize_listwise_features(features, development_rows)
    x = x[development_rows].astype(np.float64)
    y = actual_w2[development_rows].astype(np.float64)
    y_center = y - np.min(y, axis=1, keepdims=True)
    y_scale = np.quantile(y_center, 0.25, axis=1, keepdims=True)
    y_scale = np.maximum(y_scale, 1e-8)
    target = softmax(
        -y_center / (target_temperature * y_scale), axis=1
    )

    def objective(parameters: np.ndarray):
        weights = softmax(parameters)
        score = np.einsum("qcf,f->qc", x, weights)
        prediction = softmax(-score / prediction_temperature, axis=1)
        loss = -np.mean(np.sum(target * np.log(prediction + 1e-12), axis=1))
        loss += regularization * float(
            np.sum(np.square(weights - 1.0 / len(weights)))
        )
        return loss

    result = minimize(
        objective,
        np.zeros(features.shape[-1], dtype=float),
        method="L-BFGS-B",
        options={"maxiter": 100, "ftol": 1e-10},
    )
    return softmax(result.x), float(result.fun)


def cross_context_listnet_weights(
    feature_tables: list[np.ndarray],
    actual_tables: list[np.ndarray],
    development_rows: list[np.ndarray],
    *,
    target_temperature: float = 0.30,
    prediction_temperature: float = 0.50,
    context_variance_penalty: float = 0.50,
    high_frequency_indices: tuple[int, ...] = (),
    high_frequency_penalty: float = 0.10,
    regularization: float = 0.02,
) -> tuple[np.ndarray, dict]:
    """Fit shared weights with balanced cross-context ListNet risk.

    Each context contributes equally irrespective of query count. The variance
    term discourages solutions that fit only one cellular background, while the
    explicit high-frequency penalty counters the unstable high-band collapse
    observed in the first SpectraNav MVP.
    """
    prepared = []
    for features, actual, rows in zip(
        feature_tables, actual_tables, development_rows
    ):
        normalized, _ = normalize_listwise_features(features, rows)
        x = normalized[rows].astype(np.float64)
        y = actual[rows].astype(np.float64)
        y_center = y - np.min(y, axis=1, keepdims=True)
        y_scale = np.quantile(y_center, 0.25, axis=1, keepdims=True)
        y_scale = np.maximum(y_scale, 1e-8)
        target = softmax(
            -y_center / (target_temperature * y_scale), axis=1
        )
        prepared.append((x, target))

    def losses(parameters: np.ndarray):
        weights = softmax(parameters)
        values = []
        for x, target in prepared:
            score = np.einsum("qcf,f->qc", x, weights)
            prediction = softmax(
                -score / prediction_temperature, axis=1
            )
            values.append(
                -np.mean(
                    np.sum(
                        target * np.log(prediction + 1e-12), axis=1
                    )
                )
            )
        values = np.asarray(values)
        objective = float(values.mean())
        objective += context_variance_penalty * float(values.std())
        if high_frequency_indices:
            objective += high_frequency_penalty * float(
                weights[list(high_frequency_indices)].sum()
            )
        objective += regularization * float(
            np.sum(np.square(weights - 1.0 / len(weights)))
        )
        return objective, values

    result = minimize(
        lambda parameters: losses(parameters)[0],
        np.zeros(feature_tables[0].shape[-1], dtype=float),
        method="L-BFGS-B",
        options={"maxiter": 120, "ftol": 1e-10},
    )
    weights = softmax(result.x)
    objective, context_losses = losses(result.x)
    return weights, {
        "objective": float(objective),
        "context_losses": context_losses.tolist(),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def cross_context_dro_listnet_weights(
    feature_tables: list[np.ndarray],
    actual_tables: list[np.ndarray],
    development_rows: list[np.ndarray],
    *,
    target_temperature: float = 0.30,
    prediction_temperature: float = 0.50,
    dro_temperature: float = 0.10,
    high_frequency_indices: tuple[int, ...] = (),
    high_frequency_penalty: float = 0.10,
    regularization: float = 0.02,
) -> tuple[np.ndarray, dict]:
    """Fit shared weights by minimizing a smooth worst-context ListNet risk."""
    prepared = []
    for features, actual, rows in zip(
        feature_tables, actual_tables, development_rows
    ):
        normalized, _ = normalize_listwise_features(features, rows)
        x = normalized[rows].astype(np.float64)
        y = actual[rows].astype(np.float64)
        y_center = y - np.min(y, axis=1, keepdims=True)
        y_scale = np.quantile(y_center, 0.25, axis=1, keepdims=True)
        y_scale = np.maximum(y_scale, 1e-8)
        target = softmax(
            -y_center / (target_temperature * y_scale), axis=1
        )
        prepared.append((x, target))

    def losses(parameters: np.ndarray):
        weights = softmax(parameters)
        values = []
        for x, target in prepared:
            score = np.einsum("qcf,f->qc", x, weights)
            prediction = softmax(
                -score / prediction_temperature, axis=1
            )
            values.append(
                -np.mean(
                    np.sum(
                        target * np.log(prediction + 1e-12), axis=1
                    )
                )
            )
        values = np.asarray(values)
        objective = float(
            dro_temperature
            * logsumexp(values / dro_temperature)
        )
        if high_frequency_indices:
            objective += high_frequency_penalty * float(
                weights[list(high_frequency_indices)].sum()
            )
        objective += regularization * float(
            np.sum(np.square(weights - 1.0 / len(weights)))
        )
        return objective, values

    result = minimize(
        lambda parameters: losses(parameters)[0],
        np.zeros(feature_tables[0].shape[-1], dtype=float),
        method="L-BFGS-B",
        options={"maxiter": 150, "ftol": 1e-10},
    )
    weights = softmax(result.x)
    objective, context_losses = losses(result.x)
    return weights, {
        "objective": float(objective),
        "context_losses": context_losses.tolist(),
        "worst_context_loss": float(context_losses.max()),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def select_with_weights(
    features: np.ndarray,
    weights: np.ndarray,
    development_rows: np.ndarray,
    test_rows: np.ndarray,
) -> np.ndarray:
    normalized, _ = normalize_listwise_features(features, development_rows)
    scores = np.einsum("qcf,f->qc", normalized[test_rows], weights)
    return np.argmin(scores, axis=1)


def graph_smooth(
    x: np.ndarray,
    adjacency: sparse.csr_matrix,
    beta: float,
    layers: int,
) -> np.ndarray:
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degrees)
    inv_sqrt[degrees > 0] = 1.0 / np.sqrt(degrees[degrees > 0])
    propagation = sparse.diags(inv_sqrt) @ adjacency @ sparse.diags(inv_sqrt)
    result = np.asarray(x, dtype=np.float32)
    for _ in range(layers):
        result = (1 - beta) * result + beta * (propagation @ result.T).T
    return result


def directed_graph_smooth(
    x: np.ndarray,
    adjacency: sparse.csr_matrix,
    beta: float,
    layers: int,
    *,
    signed: bool = True,
) -> np.ndarray:
    """Propagate regulator signals to targets on a directed weighted graph.

    ``adjacency[source, target]`` stores confidence times regulatory sign.
    Row-normalisation by outgoing absolute weight keeps hub transcription
    factors from dominating solely through regulon size. Dispersion states use
    ``signed=False`` because a standard deviation has no activation sign.
    """
    graph = adjacency if signed else abs(adjacency)
    out_strength = np.asarray(abs(adjacency).sum(axis=1)).ravel()
    inverse = np.zeros_like(out_strength, dtype=np.float64)
    inverse[out_strength > 0] = 1.0 / out_strength[out_strength > 0]
    propagation = sparse.diags(inverse) @ graph
    result = np.asarray(x, dtype=np.float32)
    for _ in range(layers):
        propagated = result @ propagation
        result = (1.0 - beta) * result + beta * np.asarray(propagated)
    return np.asarray(result, dtype=np.float32)
