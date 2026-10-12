from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .budgeted_equivalence import (
    BUDGETS,
    EQUIVALENCE_FRACTIONS,
    BOOTSTRAPS,
    SEED,
    _bootstrap_mean,
    _context_paths,
    context_rank_matrices,
)


TRAIN_BUDGETS = (5, 10, 20)
TRAIN_FRACTIONS = (0.01, 0.05, 0.10)
PRIMARY_BUDGET = 10
PRIMARY_FRACTION = 0.05
WEIGHT_STEP = 0.10
FEATURES = ("Frozen v2", "Raw distance", "Observed cosine")


def weight_grid(step: float = WEIGHT_STEP) -> list[tuple[float, float, float]]:
    units = int(round(1 / step))
    return [
        (i / units, j / units, (units - i - j) / units)
        for i in range(units + 1)
        for j in range(units + 1 - i)
    ]


def fused_order(matrices: dict, weights: tuple[float, float, float]) -> np.ndarray:
    score = sum(
        weight * matrices["score_ranks"][feature]
        for weight, feature in zip(weights, FEATURES)
    )
    return np.argsort(score, axis=1)


def list_objective(matrices: dict, order: np.ndarray) -> float:
    true_rank = matrices["true_rank"]
    n = true_rank.shape[1]
    values = []
    for budget in TRAIN_BUDGETS:
        selected = order[:, : min(budget, n)]
        best = np.min(
            np.take_along_axis(true_rank, selected, axis=1), axis=1
        )
        for fraction in TRAIN_FRACTIONS:
            relevant = max(1, int(math.ceil(fraction * n)))
            values.append((best < relevant).mean())
    return float(np.mean(values))


def select_loco_weights(
    all_matrices: dict[str, dict], held_out: str
) -> tuple[tuple[float, float, float], pd.DataFrame]:
    rows = []
    development = [c for c in all_matrices if c != held_out]
    for weights in weight_grid():
        context_values = []
        for context in development:
            order = fused_order(all_matrices[context], weights)
            context_values.append(list_objective(all_matrices[context], order))
        rows.append(
            {
                "held_out_context": held_out,
                "weight_frozen_v2": weights[0],
                "weight_raw_distance": weights[1],
                "weight_observed_cosine": weights[2],
                "development_macro_list_utility": float(np.mean(context_values)),
                "development_min_context_utility": float(np.min(context_values)),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        [
            "development_macro_list_utility",
            "development_min_context_utility",
            "weight_frozen_v2",
            "weight_raw_distance",
        ],
        ascending=[False, False, False, False],
    )
    best = table.iloc[0]
    weights = (
        float(best["weight_frozen_v2"]),
        float(best["weight_raw_distance"]),
        float(best["weight_observed_cosine"]),
    )
    return weights, table


def evaluate_budgetnav(
    matrices: dict, weights: tuple[float, float, float]
) -> pd.DataFrame:
    order = fused_order(matrices, weights)
    true_rank = matrices["true_rank"]
    actual_w2 = matrices["actual_w2"]
    n = true_rank.shape[1]
    rows = []
    for query_i, query in enumerate(matrices["query_names"]):
        for budget in BUDGETS:
            selected = order[query_i, : min(budget, n)]
            best_local = selected[np.argmin(actual_w2[query_i, selected])]
            utility = 1 - true_rank[query_i, best_local] / max(n - 1, 1)
            for fraction in EQUIVALENCE_FRACTIONS:
                relevant = max(1, int(math.ceil(fraction * n)))
                rows.append(
                    {
                        "context": matrices["context"],
                        "outer_fold": int(matrices["folds"][query_i]),
                        "query": query,
                        "method": "BudgetNav LOCO",
                        "budget": budget,
                        "equivalence_fraction": fraction,
                        "equivalent_hit": float(
                            np.min(true_rank[query_i, selected]) < relevant
                        ),
                        "rank_utility": float(utility),
                        "best_actual_w2": float(actual_w2[query_i, best_local]),
                        "candidate_n": n,
                        "weight_frozen_v2": weights[0],
                        "weight_raw_distance": weights[1],
                        "weight_observed_cosine": weights[2],
                    }
                )
    return pd.DataFrame(rows)


def _summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["context", "method", "budget", "equivalence_fraction"]
    for salt, (key, data) in enumerate(table.groupby(keys, sort=False)):
        for metric in ("equivalent_hit", "rank_utility"):
            values = data[metric].to_numpy(float)
            lo, hi = _bootstrap_mean(values, SEED + 3000 + salt * 7)
            rows.append({
                **dict(zip(keys, key)),
                "metric": metric,
                "mean": float(values.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "query_n": len(values),
            })
    return pd.DataFrame(rows)


def _primary_effects(combined: pd.DataFrame) -> pd.DataFrame:
    focus = combined[
        combined["budget"].eq(PRIMARY_BUDGET)
        & combined["equivalence_fraction"].eq(PRIMARY_FRACTION)
        & combined["method"].isin(
            ["BudgetNav LOCO", "Frozen v2", "Raw distance", "Observed cosine"]
        )
    ]
    wide = focus.pivot_table(
        index=["context", "outer_fold", "query"],
        columns="method", values="equivalent_hit"
    ).reset_index()
    rows = []
    for salt, (context, data) in enumerate(wide.groupby("context")):
        for comparator in ("Frozen v2", "Raw distance", "Observed cosine"):
            delta = data["BudgetNav LOCO"].to_numpy() - data[comparator].to_numpy()
            lo, hi = _bootstrap_mean(delta, SEED + 5000 + salt * 13)
            fold_delta = (
                data.assign(delta=delta).groupby("outer_fold")["delta"].mean()
            )
            rows.append({
                "context": context,
                "comparator": comparator,
                "mean_gain": float(delta.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "positive_folds": int((fold_delta > 0).sum()),
                "nonnegative_folds": int((fold_delta >= 0).sum()),
                "query_n": len(delta),
            })
    return pd.DataFrame(rows)


def run(project_root: Path) -> None:
    output = project_root / "artifacts/budgetnav_loco_v1"
    source = output / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    matrices = {
        context: context_rank_matrices(project_root, context, cache)
        for context, cache in _context_paths(project_root).items()
    }
    predictions, tuning = [], []
    selected_rows = []
    for context in matrices:
        weights, grid = select_loco_weights(matrices, context)
        tuning.append(grid)
        predictions.append(evaluate_budgetnav(matrices[context], weights))
        selected_rows.append({
            "held_out_context": context,
            "weight_frozen_v2": weights[0],
            "weight_raw_distance": weights[1],
            "weight_observed_cosine": weights[2],
        })
        print(context, weights)
    budgetnav = pd.concat(predictions, ignore_index=True)
    baseline = pd.read_csv(
        project_root
        / "artifacts/budgeted_equivalence_v1/source_data/per_query_budget_metrics.csv"
    )
    combined = pd.concat([baseline, budgetnav], ignore_index=True, sort=False)
    summary = _summary(combined)
    effects = _primary_effects(combined)
    pd.concat(tuning, ignore_index=True).to_csv(
        source / "loco_weight_tuning.csv", index=False
    )
    pd.DataFrame(selected_rows).to_csv(
        source / "selected_loco_weights.csv", index=False
    )
    budgetnav.to_csv(source / "budgetnav_per_query.csv", index=False)
    summary.to_csv(source / "combined_budget_summary.csv", index=False)
    effects.to_csv(source / "paired_primary_effects.csv", index=False)
    protocol = {
        "model": "BudgetNav LOCO rank fusion",
        "features": list(FEATURES),
        "weight_grid_step": WEIGHT_STEP,
        "training_scheme": (
            "leave-one-context-out; no outcome from the held-out context enters "
            "fusion-weight selection"
        ),
        "training_objective": (
            "macro-average equivalent-candidate hit over budgets 5,10,20 and "
            "independent-test rank thresholds 1%,5%,10% in the other contexts"
        ),
        "primary_test_endpoint": {
            "budget": PRIMARY_BUDGET, "equivalence_fraction": PRIMARY_FRACTION
        },
        "comparators": [
            "Frozen v2", "Raw distance", "Observed cosine", "Random"
        ],
        "inference_unit": "held-out perturbation identity",
        "bootstrap_resamples": BOOTSTRAPS,
        "excluded": [
            "held-out-context tuning",
            "prospective experimental validation",
            "biological interchangeability",
        ],
    }
    (output / "PROTOCOL.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )

