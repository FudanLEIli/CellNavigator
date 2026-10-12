from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .k562_identity_holdout import identity_split
from .nested_identity_cv import N_FOLDS, outer_folds
from .norman_mvp import cosine_matrix


DISPERSION_WEIGHTS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
VARIANCE_PRIOR_CELLS = 20
SPLITS = ["train", "validation", "test"]


def squared_distance(left, right):
    result = (
        np.square(left).sum(axis=1, keepdims=True)
        + np.square(right).sum(axis=1)[None, :]
        - 2 * left @ right.T
    ) / left.shape[1]
    return np.maximum(result, 0.0)


def distribution_frames(cache_dir: Path, context: str):
    prefix = context.lower()
    z = np.load(cache_dir / f"{prefix}_gaussian_summaries.npz")
    meta = pd.read_csv(cache_dir / f"{prefix}_distribution_metadata.csv")
    means, variances = z["means"], z["variances"]
    frames = {}
    for split in SPLITS:
        idx = meta.index[meta.split.eq(split)].to_numpy()
        names = meta.loc[idx, "perturbation"].astype(str).to_numpy()
        control = np.flatnonzero(names == "non-targeting")[0]
        keep = names != "non-targeting"
        split_variance = variances[idx]
        split_counts = z["counts"][idx].astype(float)
        control_variance = split_variance[control]
        shrunk_variance = (
            np.maximum(split_counts[:, None] - 1, 1) * split_variance
            + VARIANCE_PRIOR_CELLS * control_variance
        ) / (
            np.maximum(split_counts[:, None] - 1, 1)
            + VARIANCE_PRIOR_CELLS
        )
        mean_delta = means[idx][keep] - means[idx][control]
        dispersion_delta = (
            np.sqrt(shrunk_variance[keep])
            - np.sqrt(shrunk_variance[control])
        )
        frames[split] = {
            "names": names[keep],
            "mean": mean_delta.astype(np.float32),
            "dispersion": dispersion_delta.astype(np.float32),
        }
    return frames


def select_from_distance(distance):
    return distance.argmin(axis=1)


def run_context(frames, context):
    names = frames["train"]["names"]
    if not all(np.array_equal(names, frames[s]["names"]) for s in SPLITS):
        raise ValueError(f"Perturbation order mismatch in {context}")
    identity = identity_split(names)
    candidates = np.flatnonzero(identity == "train")
    queries = np.flatnonzero(identity != "train")
    folds = outer_folds(names[queries])
    predicted_mean = squared_distance(
        frames["validation"]["mean"][queries],
        frames["train"]["mean"][candidates],
    )
    predicted_dispersion = squared_distance(
        frames["validation"]["dispersion"][queries],
        frames["train"]["dispersion"][candidates],
    )
    actual_mean = squared_distance(
        frames["test"]["mean"][queries],
        frames["test"]["mean"][candidates],
    )
    actual_dispersion = squared_distance(
        frames["test"]["dispersion"][queries],
        frames["test"]["dispersion"][candidates],
    )
    actual_w2 = np.sqrt(actual_mean + actual_dispersion)
    predicted_cosine = cosine_matrix(
        frames["validation"]["mean"][queries],
        frames["train"]["mean"][candidates],
    )
    actual_cosine = cosine_matrix(
        frames["test"]["mean"][queries],
        frames["test"]["mean"][candidates],
    )
    action_rows, fold_rows, tuning_rows = [], [], []
    for fold in range(N_FOLDS):
        test_local = np.flatnonzero(folds == fold)
        development = np.flatnonzero(folds != fold)
        tuning = []
        for weight in DISPERSION_WEIGHTS:
            chosen = select_from_distance(
                predicted_mean[development]
                + weight * predicted_dispersion[development]
            )
            utility = actual_w2[
                development, chosen
            ]
            tuning.append({
                "dispersion_weight": weight,
                "mean_actual_w2": utility.mean(),
            })
        tuning = pd.DataFrame(tuning).sort_values(
            ["mean_actual_w2", "dispersion_weight"],
            ascending=[True, True],
        )
        best_weight = float(tuning.iloc[0].dispersion_weight)
        tuning.insert(0, "outer_fold", fold)
        tuning.insert(0, "context", context)
        tuning_rows.append(tuning)
        chosen_distribution = select_from_distance(
            predicted_mean[test_local]
            + best_weight * predicted_dispersion[test_local]
        )
        chosen_mean_only = select_from_distance(predicted_mean[test_local])
        chosen_observed = np.argmax(predicted_cosine[test_local], axis=1)
        for row_i, query_local in enumerate(test_local):
            dist_candidate = chosen_distribution[row_i]
            mean_candidate = chosen_mean_only[row_i]
            observed_candidate = chosen_observed[row_i]
            action_rows.append({
                "context": context, "outer_fold": fold,
                "perturbation": names[queries[query_local]],
                "distribution_recommended": names[candidates[dist_candidate]],
                "mean_only_recommended": names[candidates[mean_candidate]],
                "observed_recommended": names[candidates[observed_candidate]],
                "selected_dispersion_weight": best_weight,
                "distribution_actual_w2": actual_w2[
                    query_local, dist_candidate
                ],
                "observed_actual_w2": actual_w2[
                    query_local, observed_candidate
                ],
                "mean_only_actual_w2": actual_w2[
                    query_local, mean_candidate
                ],
                "distribution_vs_mean_w2_improvement": actual_w2[
                    query_local, mean_candidate
                ] - actual_w2[query_local, dist_candidate],
                "w2_improvement": actual_w2[
                    query_local, observed_candidate
                ] - actual_w2[query_local, dist_candidate],
                "distribution_actual_mean_mse": actual_mean[
                    query_local, dist_candidate
                ],
                "observed_actual_mean_mse": actual_mean[
                    query_local, observed_candidate
                ],
                "mean_mse_improvement": actual_mean[
                    query_local, observed_candidate
                ] - actual_mean[query_local, dist_candidate],
                "distribution_actual_dispersion_mse": actual_dispersion[
                    query_local, dist_candidate
                ],
                "observed_actual_dispersion_mse": actual_dispersion[
                    query_local, observed_candidate
                ],
                "dispersion_mse_improvement": actual_dispersion[
                    query_local, observed_candidate
                ] - actual_dispersion[query_local, dist_candidate],
                "distribution_actual_mean_cosine": actual_cosine[
                    query_local, dist_candidate
                ],
                "observed_actual_mean_cosine": actual_cosine[
                    query_local, observed_candidate
                ],
                "mean_cosine_delta": actual_cosine[
                    query_local, dist_candidate
                ] - actual_cosine[query_local, observed_candidate],
            })
        fold_data = pd.DataFrame(action_rows)
        fold_data = fold_data[
            (fold_data.context == context) & (fold_data.outer_fold == fold)
        ]
        fold_rows.append({
            "context": context, "outer_fold": fold,
            "test_n": len(test_local),
            "selected_dispersion_weight": best_weight,
            "observed_actual_w2": fold_data.observed_actual_w2.mean(),
            "distribution_actual_w2": fold_data.distribution_actual_w2.mean(),
            "mean_only_actual_w2": fold_data.mean_only_actual_w2.mean(),
            "mean_w2_improvement": fold_data.w2_improvement.mean(),
            "distribution_vs_mean_w2_improvement": (
                fold_data.distribution_vs_mean_w2_improvement.mean()
            ),
            "observed_mean_cosine": fold_data.observed_actual_mean_cosine.mean(),
            "distribution_mean_cosine": (
                fold_data.distribution_actual_mean_cosine.mean()
            ),
            "mean_cosine_delta": fold_data.mean_cosine_delta.mean(),
        })
    actions = pd.DataFrame(action_rows)
    folds_table = pd.DataFrame(fold_rows)
    tuning = pd.concat(tuning_rows, ignore_index=True)
    rng = np.random.default_rng(20260724 + (2001 if context == "K562" else 2002))
    w2_delta = actions.w2_improvement.to_numpy()
    distribution_ablation_delta = (
        actions.distribution_vs_mean_w2_improvement.to_numpy()
    )
    cosine_delta = actions.mean_cosine_delta.to_numpy()
    mean_component_delta = actions.mean_mse_improvement.to_numpy()
    dispersion_component_delta = actions.dispersion_mse_improvement.to_numpy()
    distribution_switch = (
        actions.distribution_recommended != actions.mean_only_recommended
    ).to_numpy()
    geometry_switch = (
        actions.distribution_recommended != actions.observed_recommended
    ).to_numpy()
    w2_boot = np.array([
        rng.choice(w2_delta, len(w2_delta), replace=True).mean()
        for _ in range(10000)
    ])
    ablation_boot = np.array([
        rng.choice(
            distribution_ablation_delta, len(distribution_ablation_delta),
            replace=True,
        ).mean()
        for _ in range(10000)
    ])
    cosine_boot = np.array([
        rng.choice(cosine_delta, len(cosine_delta), replace=True).mean()
        for _ in range(10000)
    ])
    summary = {
        "context": context, "query_count": len(actions),
        "candidate_count": len(candidates),
        "observed_actual_w2": actions.observed_actual_w2.mean(),
        "distribution_actual_w2": actions.distribution_actual_w2.mean(),
        "mean_w2_improvement": w2_delta.mean(),
        "w2_ci_low": np.quantile(w2_boot, .025),
        "w2_ci_high": np.quantile(w2_boot, .975),
        "w2_wilcoxon_p": float(wilcoxon(w2_delta).pvalue),
        "mean_only_actual_w2": actions.mean_only_actual_w2.mean(),
        "distribution_vs_mean_w2_improvement": (
            distribution_ablation_delta.mean()
        ),
        "distribution_switch_rate": distribution_switch.mean(),
        "distribution_gain_when_switched": (
            distribution_ablation_delta[distribution_switch].mean()
            if distribution_switch.any() else 0.0
        ),
        "geometry_switch_rate": geometry_switch.mean(),
        "geometry_gain_when_switched": (
            w2_delta[geometry_switch].mean() if geometry_switch.any() else 0.0
        ),
        "ablation_ci_low": np.quantile(ablation_boot, .025),
        "ablation_ci_high": np.quantile(ablation_boot, .975),
        "ablation_wilcoxon_p": float(
            wilcoxon(distribution_ablation_delta).pvalue
        ),
        "positive_ablation_fold_count": int(
            (folds_table.distribution_vs_mean_w2_improvement > 0).sum()
        ),
        "observed_mean_cosine": actions.observed_actual_mean_cosine.mean(),
        "distribution_mean_cosine": (
            actions.distribution_actual_mean_cosine.mean()
        ),
        "mean_cosine_delta": cosine_delta.mean(),
        "cosine_ci_low": np.quantile(cosine_boot, .025),
        "cosine_ci_high": np.quantile(cosine_boot, .975),
        "cosine_wilcoxon_p": float(wilcoxon(cosine_delta).pvalue),
        "mean_mse_improvement": mean_component_delta.mean(),
        "dispersion_mse_improvement": dispersion_component_delta.mean(),
        "positive_w2_fold_count": int(
            (folds_table.mean_w2_improvement > 0).sum()
        ),
        "nonzero_dispersion_fold_count": int(
            (folds_table.selected_dispersion_weight > 0).sum()
        ),
    }
    return actions, folds_table, tuning, summary


def run(project_root: Path):
    out = project_root / "artifacts/distributional_state"
    source = out / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    summaries = []
    for context in ["K562", "RPE1"]:
        actions, folds, tuning, summary = run_context(
            distribution_frames(cache, context), context
        )
        actions.to_csv(
            source / f"{context.lower()}_distribution_actions.csv", index=False
        )
        folds.to_csv(
            source / f"{context.lower()}_distribution_folds.csv", index=False
        )
        tuning.to_csv(
            source / f"{context.lower()}_dispersion_tuning.csv", index=False
        )
        summaries.append(summary)
    summary = pd.DataFrame(summaries)
    summary.to_csv(source / "distribution_context_summary.csv", index=False)
    protocol = {
        "state_model": "diagonal Gaussian over 2,000 prespecified genes",
        "mean_state": "perturbation mean minus split-matched control mean",
        "dispersion_state": "perturbation standard deviation minus split-matched control standard deviation",
        "variance_shrinkage": f"{VARIANCE_PRIOR_CELLS} control-equivalent prior cells within each split",
        "distance": "per-gene diagonal Gaussian Wasserstein-2",
        "dispersion_weight_grid": DISPERSION_WEIGHTS,
        "validation": "five outer identity folds; dispersion weight selected on non-test identities",
        "primary_comparator": "Observed centroid-cosine retrieval",
        "held_out_identity_outcomes_used_for_weight_selection": False,
        "tuning_outcome": (
            "test-replicate W2 from non-test development identities; "
            "outer-fold identities remain untouched"
        ),
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    lines = ["# Distribution-aware intervention ranking", ""]
    for row in summary.itertuples(index=False):
        lines.extend([
            f"## {row.context}", "",
            f"- Queries/candidates: {row.query_count}/{row.candidate_count}.",
            f"- Observed versus distribution-selected test W2: "
            f"{row.observed_actual_w2:.3f} versus "
            f"{row.distribution_actual_w2:.3f}.",
            f"- W2 improvement: {row.mean_w2_improvement:.3f} "
            f"(95% CI {row.w2_ci_low:.3f} to {row.w2_ci_high:.3f}; "
            f"Wilcoxon P={row.w2_wilcoxon_p:.3g}).",
            f"- Mean-only versus full-distribution test W2: "
            f"{row.mean_only_actual_w2:.3f} versus "
            f"{row.distribution_actual_w2:.3f}; distribution-specific "
            f"improvement {row.distribution_vs_mean_w2_improvement:.3f} "
            f"(95% CI {row.ablation_ci_low:.3f} to "
            f"{row.ablation_ci_high:.3f}; P={row.ablation_wilcoxon_p:.3g}).",
            f"- Independent mean-cosine delta: {row.mean_cosine_delta:.3f} "
            f"(95% CI {row.cosine_ci_low:.3f} to "
            f"{row.cosine_ci_high:.3f}; P={row.cosine_wilcoxon_p:.3g}).",
            f"- W2 component improvements (mean MSE / dispersion MSE): "
            f"{row.mean_mse_improvement:.3f} / "
            f"{row.dispersion_mse_improvement:.3f}.",
            f"- Positive W2 folds: {row.positive_w2_fold_count}/5; "
            f"positive distribution-ablation folds: "
            f"{row.positive_ablation_fold_count}/5; "
            f"non-zero dispersion weight: "
            f"{row.nonzero_dispersion_fold_count}/5 folds.",
            "",
        ])
    (out / "DISTRIBUTIONAL_RANKING_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n".join(lines))
