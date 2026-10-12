from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .distributional_ranking import distribution_frames, squared_distance
from .k562_identity_holdout import identity_split
from .nested_identity_cv import outer_folds
from .norman_mvp import cosine_matrix


BUDGETS = (1, 2, 5, 10, 20, 50)
EQUIVALENCE_FRACTIONS = (0.01, 0.05, 0.10)
METHODS = ("Frozen v2", "Raw distance", "Observed cosine", "Random")
BOOTSTRAPS = 5000
SEED = 20260725


def _random_hit_probability(n: int, relevant_n: int, budget: int) -> float:
    """Probability that a size-budget sample contains >=1 relevant item."""
    budget = min(budget, n)
    if budget <= 0 or relevant_n <= 0:
        return 0.0
    if budget > n - relevant_n:
        return 1.0
    log_miss = (
        math.lgamma(n - relevant_n + 1)
        - math.lgamma(budget + 1)
        - math.lgamma(n - relevant_n - budget + 1)
        - math.lgamma(n + 1)
        + math.lgamma(budget + 1)
        + math.lgamma(n - budget + 1)
    )
    return float(1.0 - math.exp(log_miss))


def _random_rank_utility(n: int, budget: int) -> float:
    """Expected 1-normalized best ordinal rank under sampling without replacement."""
    budget = min(budget, n)
    if n <= 1:
        return 1.0
    expected_min_rank = (n - budget) / (budget + 1)
    return float(1.0 - expected_min_rank / (n - 1))


def _bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _context_paths(project_root: Path) -> dict[str, Path]:
    return {
        "K562": project_root / "artifacts/distributional_state/cache",
        "RPE1": project_root / "artifacts/distributional_state/cache",
        "HepG2": project_root / "artifacts/external_validation_hepg2_v1/cache",
        "Jurkat": project_root / "artifacts/external_validation_jurkat_v2/cache",
    }


def _load_transform(project_root: Path):
    scripts = project_root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from benchmark_v2_state_scaling import transform_frames

    return transform_frames


def _weights(project_root: Path, context: str, variant: str) -> dict[int, float]:
    if context == "Jurkat":
        path = (
            project_root
            / "artifacts/external_validation_jurkat_v2/locked_results"
            / f"{variant}_tuning.csv"
        )
        score = "development_raw_w2"
    else:
        path = project_root / "artifacts/v2_state_scaling/source_data/all_inner_tuning.csv"
        score = "development_raw_w2"
    table = pd.read_csv(path)
    if "context" in table:
        table = table[table["context"].eq(context)]
    if "variant" in table:
        table = table[table["variant"].eq(variant)]
    best = (
        table.sort_values([score, "dispersion_weight"])
        .groupby("outer_fold", as_index=False)
        .first()
    )
    return dict(zip(best["outer_fold"].astype(int), best["dispersion_weight"]))


def context_rank_matrices(project_root: Path, context: str, cache: Path):
    transform_frames = _load_transform(project_root)
    raw = distribution_frames(cache, context)
    scaled = transform_frames(raw, cache, context, "control_z")
    names = raw["train"]["names"]
    labels = identity_split(names)
    candidates = np.flatnonzero(labels == "train")
    queries = np.flatnonzero(labels != "train")
    folds = outer_folds(names[queries])

    def components(frames):
        mean = squared_distance(
            frames["validation"]["mean"][queries],
            frames["train"]["mean"][candidates],
        )
        dispersion = squared_distance(
            frames["validation"]["dispersion"][queries],
            frames["train"]["dispersion"][candidates],
        )
        return mean, dispersion

    scaled_mean, scaled_dispersion = components(scaled)
    raw_mean, raw_dispersion = components(raw)
    observed_cosine = cosine_matrix(
        raw["validation"]["mean"][queries],
        raw["train"]["mean"][candidates],
    )
    actual_mean = squared_distance(
        raw["test"]["mean"][queries], raw["test"]["mean"][candidates]
    )
    actual_dispersion = squared_distance(
        raw["test"]["dispersion"][queries], raw["test"]["dispersion"][candidates]
    )
    actual_w2 = np.sqrt(actual_mean + actual_dispersion)

    v2_weights = _weights(project_root, context, "control_z")
    raw_weights = _weights(project_root, context, "raw")
    ranking = {}
    v2_score = np.empty_like(scaled_mean)
    raw_score = np.empty_like(raw_mean)
    for fold in range(5):
        rows = folds == fold
        v2_score[rows] = scaled_mean[rows] + v2_weights[fold] * scaled_dispersion[rows]
        raw_score[rows] = raw_mean[rows] + raw_weights[fold] * raw_dispersion[rows]
    ranking["Frozen v2"] = np.argsort(v2_score, axis=1)
    ranking["Raw distance"] = np.argsort(raw_score, axis=1)
    ranking["Observed cosine"] = np.argsort(-observed_cosine, axis=1)

    true_order = np.argsort(actual_w2, axis=1)
    true_rank = np.empty_like(true_order)
    true_rank[np.arange(len(queries))[:, None], true_order] = np.arange(len(candidates))
    score_ranks = {}
    for method, order in ranking.items():
        inverse = np.empty_like(order)
        inverse[np.arange(len(queries))[:, None], order] = np.arange(len(candidates))
        score_ranks[method] = inverse.astype(np.float32) / max(len(candidates) - 1, 1)
    return {
        "context": context,
        "query_names": names[queries],
        "candidate_names": names[candidates],
        "folds": folds,
        "ranking": ranking,
        "score_ranks": score_ranks,
        "true_rank": true_rank,
        "actual_w2": actual_w2,
        "raw_score": raw_score,
        "raw_mean_score": raw_mean,
        "raw_dispersion_score": raw_dispersion,
        "scaled_mean_score": scaled_mean,
        "scaled_dispersion_score": scaled_dispersion,
        "observed_cosine_score": observed_cosine,
        "query_validation_mean": raw["validation"]["mean"][queries],
        "query_validation_dispersion": raw["validation"]["dispersion"][queries],
    }


def evaluate_context(project_root: Path, context: str, cache: Path):
    matrices = context_rank_matrices(project_root, context, cache)
    context = matrices["context"]
    query_names = matrices["query_names"]
    candidate_names = matrices["candidate_names"]
    folds = matrices["folds"]
    ranking = matrices["ranking"]
    true_rank = matrices["true_rank"]
    actual_w2 = matrices["actual_w2"]
    n_candidates = len(candidate_names)
    rows = []
    retrieval = []
    for query_i, query_name in enumerate(query_names):
        for method in METHODS:
            for budget in BUDGETS:
                k = min(budget, n_candidates)
                if method == "Random":
                    rank_utility = _random_rank_utility(n_candidates, k)
                    best_w2 = np.nan
                else:
                    selected = ranking[method][query_i, :k]
                    best_local = selected[np.argmin(actual_w2[query_i, selected])]
                    rank_utility = 1.0 - true_rank[query_i, best_local] / max(
                        n_candidates - 1, 1
                    )
                    best_w2 = actual_w2[query_i, best_local]
                for fraction in EQUIVALENCE_FRACTIONS:
                    relevant_n = max(1, int(math.ceil(fraction * n_candidates)))
                    if method == "Random":
                        hit = _random_hit_probability(n_candidates, relevant_n, k)
                    else:
                        hit = float(np.min(true_rank[query_i, ranking[method][query_i, :k]]) < relevant_n)
                    rows.append(
                        {
                            "context": context,
                            "outer_fold": int(folds[query_i]),
                            "query": query_name,
                            "method": method,
                            "budget": budget,
                            "equivalence_fraction": fraction,
                            "equivalent_hit": hit,
                            "rank_utility": rank_utility,
                            "best_actual_w2": best_w2,
                            "candidate_n": n_candidates,
                        }
                    )
        for predicted_position, candidate_local in enumerate(
            ranking["Frozen v2"][query_i, : min(50, n_candidates)], start=1
        ):
            retrieval.append(
                {
                    "context": context,
                    "outer_fold": int(folds[query_i]),
                    "query": query_name,
                    "candidate": candidate_names[candidate_local],
                    "predicted_rank": predicted_position,
                    "actual_rank": int(true_rank[query_i, candidate_local] + 1),
                    "actual_rank_percentile": float(
                        (true_rank[query_i, candidate_local] + 1) / n_candidates
                    ),
                    "actual_w2": float(actual_w2[query_i, candidate_local]),
                    "equivalent_top_5pct": bool(
                        true_rank[query_i, candidate_local]
                        < max(1, int(math.ceil(0.05 * n_candidates)))
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(retrieval)


def summarize(per_query: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["context", "method", "budget", "equivalence_fraction"]
    for salt, (key, table) in enumerate(per_query.groupby(keys, sort=False)):
        for metric in ("equivalent_hit", "rank_utility"):
            values = table[metric].to_numpy(float)
            lo, hi = _bootstrap_mean(values, SEED + salt * 7 + (metric == "rank_utility"))
            rows.append(
                {
                    **dict(zip(keys, key)),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "query_n": len(values),
                }
            )
    return pd.DataFrame(rows)


def fold_effects(per_query: pd.DataFrame) -> pd.DataFrame:
    focus = per_query[
        per_query["budget"].eq(10)
        & per_query["equivalence_fraction"].eq(0.05)
        & per_query["method"].isin(["Frozen v2", "Observed cosine", "Random"])
    ]
    wide = focus.pivot_table(
        index=["context", "outer_fold", "query"],
        columns="method",
        values="equivalent_hit",
    ).reset_index()
    fold = wide.groupby(["context", "outer_fold"], as_index=False).agg(
        frozen_v2=("Frozen v2", "mean"),
        observed_cosine=("Observed cosine", "mean"),
        random=("Random", "mean"),
        query_n=("query", "size"),
    )
    fold["gain_vs_observed"] = fold["frozen_v2"] - fold["observed_cosine"]
    fold["enrichment_vs_random"] = fold["frozen_v2"] / fold["random"]
    return fold


def paired_primary_effects(per_query: pd.DataFrame) -> pd.DataFrame:
    focus = per_query[
        per_query["budget"].eq(10)
        & per_query["equivalence_fraction"].eq(0.05)
        & per_query["method"].isin(["Frozen v2", "Observed cosine", "Random"])
    ]
    wide = focus.pivot_table(
        index=["context", "outer_fold", "query"],
        columns="method",
        values="equivalent_hit",
    ).reset_index()
    rows = []
    for salt, (context, table) in enumerate(wide.groupby("context")):
        for comparator in ("Observed cosine", "Random"):
            delta = (
                table["Frozen v2"].to_numpy(float)
                - table[comparator].to_numpy(float)
            )
            lo, hi = _bootstrap_mean(delta, SEED + 900 + salt * 11)
            rows.append(
                {
                    "context": context,
                    "comparator": comparator,
                    "mean_gain": float(delta.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "query_n": len(delta),
                    "positive_folds": int(
                        (
                            table.assign(delta=delta)
                            .groupby("outer_fold")["delta"]
                            .mean()
                            > 0
                        ).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def run(project_root: Path) -> None:
    output = project_root / "artifacts/budgeted_equivalence_v1"
    source = output / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    all_query, all_retrieval = [], []
    for context, cache in _context_paths(project_root).items():
        query, retrieval = evaluate_context(project_root, context, cache)
        all_query.append(query)
        all_retrieval.append(retrieval)
        print(context, len(query["query"].unique()), query["candidate_n"].iloc[0])
    per_query = pd.concat(all_query, ignore_index=True)
    retrieval = pd.concat(all_retrieval, ignore_index=True)
    summary = summarize(per_query)
    folds = fold_effects(per_query)
    effects = paired_primary_effects(per_query)
    per_query.to_csv(source / "per_query_budget_metrics.csv", index=False)
    retrieval.to_csv(source / "frozen_v2_top50_retrieval.csv", index=False)
    summary.to_csv(source / "budget_summary.csv", index=False)
    folds.to_csv(source / "fold_effects_budget10_top5pct.csv", index=False)
    effects.to_csv(source / "paired_primary_effects.csv", index=False)
    protocol = {
        "status": "retrospective budget-constrained evaluation of frozen v2",
        "contexts": list(_context_paths(project_root)),
        "budgets": list(BUDGETS),
        "functional_equivalence": (
            "candidate belongs to the top q fraction by independent-test raw "
            "diagonal-Gaussian W2 for a query; q in 1%, 5%, 10%"
        ),
        "primary_operating_point": {"budget": 10, "q": 0.05},
        "rank_utility": "1 minus normalized independent-test ordinal rank of the best selected candidate",
        "random_baseline": "exact expectation under uniform sampling without replacement",
        "frozen_model": "control-z diagonal-Gaussian ranking with previously selected fold weights",
        "inference_unit": "held-out perturbation identity",
        "confidence_intervals": f"{BOOTSTRAPS} query bootstrap resamples",
        "excluded_claims": [
            "prospective experimental validation",
            "biological interchangeability",
            "clinical actionability",
            "reachability",
        ],
    }
    (output / "PROTOCOL.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
