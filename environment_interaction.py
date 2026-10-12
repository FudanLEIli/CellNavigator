from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .distributional_ranking import distribution_frames, squared_distance
from .frangieh_a375 import CONDITIONS, V2_DISPERSION_WEIGHT, _control_z
from .k562_identity_holdout import identity_split
from .mechanism_listwise import (
    _distance,
    _evaluate,
    _list_utility,
    _project,
    _rank_score,
    build_module_matrices,
)


SHRINKAGES = (0.0, 0.1, 1.0, 10.0)
BLENDS = (0.0, 0.25, 0.50, 0.75, 1.0)
SEED = 20260728


def _query_list_utility(score: np.ndarray, actual: np.ndarray) -> np.ndarray:
    score = score.copy()
    actual = actual.copy()
    np.fill_diagonal(score, np.inf)
    np.fill_diagonal(actual, np.inf)
    order = np.argsort(score, axis=1)
    truth_order = np.argsort(actual, axis=1)
    truth_rank = np.empty_like(truth_order)
    truth_rank[np.arange(len(actual))[:, None], truth_order] = np.arange(
        actual.shape[1]
    )
    n = actual.shape[1] - 1
    values = []
    for budget in (5, 10, 20):
        selected = order[:, :budget]
        best = np.min(np.take_along_axis(truth_rank, selected, axis=1), axis=1)
        for fraction in (0.01, 0.05, 0.10):
            values.append(best < max(1, int(np.ceil(fraction * n))))
    return np.mean(values, axis=0)


def _crossfit_interaction(
    target: dict,
    sources: list[dict],
    candidates: np.ndarray,
    shrinkage: float,
    *,
    additive: bool = False,
    shuffled: bool = False,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    names = target["train"]["names"][candidates]
    from .nested_identity_cv import outer_folds

    folds = outer_folds(names)
    rng = np.random.default_rng(seed)
    result = {}
    for component in ("mean", "dispersion"):
        left = sources[0]["train"][component]
        right = sources[1]["train"][component]
        base = (left + right) / 2
        contrast = left - right
        response = target["train"][component] - base
        predicted = np.zeros_like(response[candidates])
        for fold in range(5):
            held_local = np.flatnonzero(folds == fold)
            anchor_local = np.flatnonzero(folds != fold)
            anchors = candidates[anchor_local]
            held = candidates[held_local]
            d = contrast[anchors]
            r = response[anchors]
            d_mean = d.mean(axis=0)
            intercept = r.mean(axis=0)
            dc = d - d_mean
            rc = r - intercept
            if shuffled:
                rc = rc[rng.permutation(len(rc))]
            variance = np.square(dc).sum(axis=0)
            positive = variance[variance > 0]
            scale = float(np.median(positive)) if len(positive) else 1.0
            if additive:
                slope = np.zeros_like(variance)
            else:
                slope = np.divide(
                    (dc * rc).sum(axis=0),
                    variance + shrinkage * scale,
                    out=np.zeros_like(variance),
                    where=(variance + shrinkage * scale) > 0,
                )
            predicted[held_local] = (
                base[held] + intercept + (contrast[held] - d_mean) * slope
            )
        result[component] = predicted.astype(np.float32)
    return result


def _module_score(
    target: dict, prediction: dict, query_idx: np.ndarray
) -> np.ndarray:
    return _rank_score(
        squared_distance(target["mean"][query_idx], prediction["mean"])
        + V2_DISPERSION_WEIGHT
        * squared_distance(
            target["dispersion"][query_idx], prediction["dispersion"]
        )
    )


def _select_configuration(
    gene_dev_rank: np.ndarray,
    actual_dev: np.ndarray,
    target: dict,
    sources: list[dict],
    candidates: np.ndarray,
    mode: str,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    from .nested_identity_cv import outer_folds

    rows, predictions = [], {}
    candidate_folds = outer_folds(target["train"]["names"][candidates])
    baseline_query_utility = _query_list_utility(gene_dev_rank, actual_dev)
    shrinkages = (1.0,) if mode == "additive" else SHRINKAGES
    for shrinkage in shrinkages:
        prediction = _crossfit_interaction(
            target,
            sources,
            candidates,
            shrinkage,
            additive=mode == "additive",
            shuffled=mode == "shuffled",
            seed=seed,
        )
        predictions[shrinkage] = prediction
        module_rank = _module_score(
            target["train"], prediction, candidates
        )
        for blend in BLENDS:
            score = (1 - blend) * gene_dev_rank + blend * module_rank
            query_utility = _query_list_utility(score, actual_dev)
            fold_gain = [
                float(np.mean(
                    query_utility[candidate_folds == fold]
                    - baseline_query_utility[candidate_folds == fold]
                ))
                for fold in range(5)
            ]
            rows.append({
                "mode": mode, "shrinkage": shrinkage, "blend": blend,
                "development_list_utility": _list_utility(
                    score, actual_dev, self_match=True
                ),
                "development_gain": float(
                    np.mean(query_utility - baseline_query_utility)
                ),
                "positive_folds": int(np.sum(np.asarray(fold_gain) > 0)),
            })
    table = pd.DataFrame(rows)
    table["robust_pass"] = table["blend"].eq(0) | (
        table["development_gain"].gt(0) & table["positive_folds"].ge(4)
    )
    eligible = table[table["robust_pass"]].sort_values(
        ["development_list_utility", "blend", "shrinkage"],
        ascending=[False, True, False],
    )
    best = eligible.iloc[0]
    table["selected"] = (
        table["shrinkage"].eq(best["shrinkage"])
        & table["blend"].eq(best["blend"])
    )
    return {
        "prediction": predictions[float(best["shrinkage"])],
        "blend": float(best["blend"]),
        "shrinkage": float(best["shrinkage"]),
    }, table


def run(project_root: Path) -> None:
    cache = project_root / "artifacts/frangieh_a375_frozen_v1/cache"
    output = project_root / "artifacts/a375_environment_interaction_v1"
    source_dir = output / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    gene_ids = pd.read_csv(cache / "a375_distribution_genes.csv")[
        "gene_id"
    ].astype(str).to_numpy()
    families = build_module_matrices(project_root, gene_ids)
    combined = sparse.vstack(
        [families["reactome"][0], families["regulatory"][0]], format="csr"
    )
    permutation = np.random.default_rng(SEED).permutation(len(gene_ids))
    random_combined = combined[:, permutation]
    pd.DataFrame([
        {
            "representation": "mechanism",
            "module_n": combined.shape[0],
            "gene_n": int((np.asarray(combined.getnnz(axis=0)) > 0).sum()),
        },
        {
            "representation": "random",
            "module_n": random_combined.shape[0],
            "gene_n": int((np.asarray(random_combined.getnnz(axis=0)) > 0).sum()),
        },
    ]).to_csv(source_dir / "representation_manifest.csv", index=False)

    contexts = list(CONDITIONS.values())
    result_rows, tuning_rows, promotion_rows = [], [], []
    for context_i, context in enumerate(contexts):
        source_contexts = [x for x in contexts if x != context]
        raw_target = distribution_frames(cache, context)
        z_target = _control_z(raw_target, cache, context)
        z_sources = [
            _control_z(distribution_frames(cache, x), cache, x)
            for x in source_contexts
        ]
        names = z_target["train"]["names"]
        candidates = np.flatnonzero(identity_split(names) == "train")
        queries = np.flatnonzero(identity_split(names) != "train")
        actual_dev = np.sqrt(
            squared_distance(
                raw_target["validation"]["mean"][candidates],
                raw_target["validation"]["mean"][candidates],
            )
            + squared_distance(
                raw_target["validation"]["dispersion"][candidates],
                raw_target["validation"]["dispersion"][candidates],
            )
        )
        actual_test = np.sqrt(
            squared_distance(
                raw_target["test"]["mean"][queries],
                raw_target["test"]["mean"][candidates],
            )
            + squared_distance(
                raw_target["test"]["dispersion"][queries],
                raw_target["test"]["dispersion"][candidates],
            )
        )
        gene_dev_rank = _rank_score(_distance(
            z_target, z_sources, candidates, candidates, "train", "train"
        ))
        gene_test_rank = _rank_score(_distance(
            z_target, z_sources, queries, candidates, "validation", "train"
        ))
        scores = {"Frozen v2": gene_test_rank}
        selected_utility = {}

        for representation, matrix in (
            ("mechanism", combined),
            ("random", random_combined),
        ):
            target_module = _project(z_target, matrix)
            source_modules = [_project(x, matrix) for x in z_sources]
            modes = (
                ("additive", "Additive-env gated"),
                ("interaction", "InteractionNav gated"),
                ("shuffled", "Shuffled-interaction gated"),
            ) if representation == "mechanism" else (
                ("interaction", "Random-interaction gated"),
            )
            for mode, label in modes:
                selected, table = _select_configuration(
                    gene_dev_rank, actual_dev, target_module,
                    source_modules, candidates, mode,
                    SEED + context_i * 100 + (17 if mode == "shuffled" else 0),
                )
                table.insert(0, "representation", representation)
                table.insert(0, "target_context", context)
                tuning_rows.append(table)
                test_module_rank = _module_score(
                    target_module["validation"], selected["prediction"], queries
                )
                blend = selected["blend"]
                scores[label] = (
                    (1 - blend) * gene_test_rank + blend * test_module_rank
                )
                selected_utility[label] = float(
                    table.loc[table["selected"], "development_list_utility"].iloc[0]
                )
        interaction_utility = selected_utility["InteractionNav gated"]
        control_utility = max(
            selected_utility["Additive-env gated"],
            selected_utility["Shuffled-interaction gated"],
            selected_utility["Random-interaction gated"],
            _list_utility(gene_dev_rank, actual_dev, self_match=True),
        )
        promoted = interaction_utility > control_utility
        if not promoted:
            scores["InteractionNav gated"] = gene_test_rank
        promotion_rows.append({
            "target_context": context,
            "interaction_development_utility": interaction_utility,
            "best_control_development_utility": control_utility,
            "development_margin": interaction_utility - control_utility,
            "promoted": promoted,
        })
        for method, score in scores.items():
            result_rows.extend(_evaluate(
                context, method, score, actual_test, names, queries, len(candidates)
            ))

    result = pd.DataFrame(result_rows)
    result.to_csv(source_dir / "per_query_budget_metrics.csv", index=False)
    pd.concat(tuning_rows, ignore_index=True).to_csv(
        source_dir / "development_configuration_tuning.csv", index=False
    )
    pd.DataFrame(promotion_rows).to_csv(
        source_dir / "causal_promotion_gate.csv", index=False
    )
    summary = (
        result.groupby(
            ["target_context", "method", "budget", "equivalence_fraction"],
            as_index=False,
        )
        .agg(
            equivalent_hit=("equivalent_hit", "mean"),
            rank_utility=("rank_utility", "mean"),
            query_n=("query", "nunique"),
        )
    )
    summary.to_csv(source_dir / "summary.csv", index=False)
    (output / "LOCKED_PROTOCOL.json").write_text(json.dumps({
        "test_guide_use": "evaluation only",
        "held_out_target_identity_use": "evaluation only",
        "interaction": (
            "target = source mean + environment intercept + "
            "module-wise shrinkage slope * source-environment contrast"
        ),
        "candidate_prediction": "five-fold perturbation-identity cross-fit",
        "selection": "candidate train/validation listwise budget utility",
        "stability_gate": (
            "non-zero calibration requires positive development gain in "
            "at least four of five perturbation-identity folds"
        ),
        "causal_promotion_gate": (
            "real interaction must exceed Frozen v2, additive environment, "
            "shuffled interaction and random-module interaction on development guides"
        ),
        "shrinkage_grid": SHRINKAGES, "blend_grid": BLENDS,
        "zero_calibration_rejection": True,
        "controls": [
            "additive environment effect", "shuffled interaction pairing",
            "gene-permuted random modules", "Frozen v2",
        ],
    }, indent=2), encoding="utf-8")
