from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .k562_identity_holdout import identity_split
from .k562_listwise_ranker import (
    fit_listwise_ensemble, predict_listwise,
)
from .k562_mvp import SEED, build_delta_frames
from .k562_pairwise_ranker import (
    MIXING, RISKS, choose_actions, conformal_metrics,
)
from .k562_probabilistic import pair_features
from .norman_mvp import cosine_matrix
from .rpe1_context_transfer import align_contexts, all_gene_deltas


N_FOLDS = 5


def hash_fraction(name, salt):
    digest = hashlib.sha256(f"{salt}::{name}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def outer_folds(names):
    return np.asarray([
        int(hash_fraction(name, "cellnavigator-nested-outer-20260724") * N_FOLDS)
        % N_FOLDS
        for name in names.astype(str)
    ])


def development_roles(names, outer_fold):
    roles = []
    salt = f"cellnavigator-nested-inner-20260724-fold{outer_fold}"
    for name in names.astype(str):
        value = hash_fraction(name, salt)
        roles.append("fit" if value < .60 else ("tune" if value < .80 else "calibrate"))
    return np.asarray(roles)


def prepare_k562(project_root):
    base = project_root / "artifacts/k562_mvp/cache"
    z = np.load(base / "k562_centroids_all_genes.npz")
    metadata = pd.read_csv(base / "k562_centroid_metadata.csv")
    genes = pd.read_csv(base / "k562_genes.csv")
    frames, _ = build_delta_frames(z["centroids"], metadata, genes)
    return frames["train"]["names"], frames


def prepare_rpe1(project_root):
    kbase = project_root / "artifacts/k562_mvp/cache"
    kz = np.load(kbase / "k562_centroids_all_genes.npz")
    kmeta = pd.read_csv(kbase / "k562_centroid_metadata.csv")
    kgenes = pd.read_csv(kbase / "k562_genes.csv")
    rbase = project_root / "artifacts/rpe1_context_transfer/cache"
    rz = np.load(rbase / "rpe1_centroids_all_genes.npz")
    rmeta = pd.read_csv(rbase / "rpe1_centroid_metadata.csv")
    rgenes = pd.read_csv(rbase / "rpe1_genes.csv")
    aligned = align_contexts(
        all_gene_deltas(kz["centroids"], kmeta), kgenes,
        all_gene_deltas(rz["centroids"], rmeta), rgenes,
    )
    return aligned["names"], aligned["rpe1"]


def run_context(names, frames, context):
    identities = identity_split(names)
    candidate_idx = np.flatnonzero(identities == "train")
    reliability = np.diag(cosine_matrix(
        frames["train"], frames["validation"]
    ))
    # Strict identity holdout: eligibility cannot depend on a held-out
    # identity's own historical response. Replicate consistency is retained
    # only for prespecified stratified reporting.
    query_idx = np.flatnonzero(identities != "train")
    folds = outer_folds(names[query_idx])
    features = pair_features(
        frames["validation"][query_idx],
        frames["train"][candidate_idx],
        frames["validation"][candidate_idx],
    )
    observed_score = features[:, :, 0]
    actual = cosine_matrix(
        frames["test"][query_idx], frames["test"][candidate_idx]
    )
    self_similarity = reliability[candidate_idx]
    thresholds = {
        q: float(np.quantile(self_similarity, q)) for q in [.10, .25, .50]
    }
    action_rows, fold_rows, tuning_rows, conformal_rows = [], [], [], []
    for fold in range(N_FOLDS):
        outer_test = np.flatnonzero(folds == fold)
        development = np.flatnonzero(folds != fold)
        roles = development_roles(names[query_idx[development]], fold)
        role_idx = {
            role: development[roles == role]
            for role in ["fit", "tune", "calibrate"]
        }
        if min(map(len, role_idx.values())) < 5 or len(outer_test) < 5:
            raise ValueError(f"Insufficient identities in {context} fold {fold}")
        models = fit_listwise_ensemble(
            features[role_idx["fit"]], actual[role_idx["fit"]],
            n_models=8,
        )
        predict_idx = np.concatenate([
            role_idx["tune"], role_idx["calibrate"], outer_test
        ])
        learned, uncertainty = predict_listwise(features[predict_idx], models)
        lengths = [len(role_idx["tune"]), len(role_idx["calibrate"])]
        cuts = np.cumsum([0] + lengths)
        tune_slice = slice(cuts[0], cuts[1])
        calibration_slice = slice(cuts[1], cuts[2])
        test_slice = slice(cuts[2], None)
        tuning = []
        for mixing in MIXING:
            for risk in RISKS:
                selected = choose_actions(
                    names, query_idx[role_idx["tune"]], candidate_idx,
                    observed_score[role_idx["tune"]], learned[tune_slice],
                    uncertainty[tune_slice], actual[role_idx["tune"]],
                    mixing, risk,
                )
                tuning.append({
                    "mixing": mixing, "risk": risk,
                    "mean_actual_utility": selected.actual_utility.mean(),
                })
        tuning = pd.DataFrame(tuning).sort_values(
            ["mean_actual_utility", "mixing", "risk"],
            ascending=[False, True, True],
        )
        best = tuning.iloc[0]
        tuning.insert(0, "outer_fold", fold)
        tuning.insert(0, "context", context)
        tuning_rows.append(tuning)
        calibration = choose_actions(
            names, query_idx[role_idx["calibrate"]], candidate_idx,
            observed_score[role_idx["calibrate"]],
            learned[calibration_slice], uncertainty[calibration_slice],
            actual[role_idx["calibrate"]], float(best.mixing), float(best.risk),
        )
        reranked = choose_actions(
            names, query_idx[outer_test], candidate_idx,
            observed_score[outer_test], learned[test_slice],
            uncertainty[test_slice], actual[outer_test],
            float(best.mixing), float(best.risk),
        )
        observed = choose_actions(
            names, query_idx[outer_test], candidate_idx,
            observed_score[outer_test], learned[test_slice],
            uncertainty[test_slice], actual[outer_test], 0.0, 0.0,
        )
        joined = reranked.set_index("perturbation").join(
            observed.set_index("perturbation")[["recommended", "actual_utility"]],
            rsuffix="_observed",
        ).reset_index()
        joined["delta"] = (
            joined.actual_utility - joined.actual_utility_observed
        )
        joined["context"] = context
        joined["outer_fold"] = fold
        joined["train_validation_reliability"] = reliability[
            query_idx[outer_test]
        ]
        joined["reliability_stratum"] = np.where(
            joined.train_validation_reliability >= .30,
            "reliable_ge_0.30", "all_other",
        )
        action_rows.append(joined)
        bounds = conformal_metrics(calibration, reranked, thresholds)
        bounds.insert(0, "outer_fold", fold)
        bounds.insert(0, "context", context)
        conformal_rows.append(bounds)
        fold_rows.append({
            "context": context, "outer_fold": fold,
            "test_n": len(outer_test),
            "fit_n": len(role_idx["fit"]),
            "tune_n": len(role_idx["tune"]),
            "calibrate_n": len(role_idx["calibrate"]),
            "selected_mixing": float(best.mixing),
            "selected_risk": float(best.risk),
            "observed_utility": observed.actual_utility.mean(),
            "reranked_utility": reranked.actual_utility.mean(),
            "mean_delta": joined.delta.mean(),
            "improved_fraction": (joined.delta > 0).mean(),
            "worsened_fraction": (joined.delta < 0).mean(),
            "unchanged_fraction": (joined.delta == 0).mean(),
        })
    actions = pd.concat(action_rows, ignore_index=True)
    folds_table = pd.DataFrame(fold_rows)
    tuning = pd.concat(tuning_rows, ignore_index=True)
    conformal = pd.concat(conformal_rows, ignore_index=True)
    conformal = conformal.merge(
        folds_table[["outer_fold", "test_n"]], on="outer_fold", how="left"
    )
    conformal["covered_n_estimate"] = (
        conformal.empirical_bound_coverage * conformal.test_n
    )
    conformal["accepted_success_n_estimate"] = (
        conformal.accepted_success_fraction.fillna(0) * conformal.accepted_n
    )
    aggregate_conformal = (
        conformal.groupby(["context", "alpha", "threshold_quantile"], as_index=False)
        .agg(
            support_threshold=("support_threshold", "first"),
            test_n=("test_n", "sum"),
            covered_n=("covered_n_estimate", "sum"),
            accepted_n=("accepted_n", "sum"),
            accepted_success_n=("accepted_success_n_estimate", "sum"),
        )
    )
    aggregate_conformal["empirical_bound_coverage"] = (
        aggregate_conformal.covered_n / aggregate_conformal.test_n
    )
    aggregate_conformal["accepted_fraction"] = (
        aggregate_conformal.accepted_n / aggregate_conformal.test_n
    )
    aggregate_conformal["accepted_success_fraction"] = np.divide(
        aggregate_conformal.accepted_success_n,
        aggregate_conformal.accepted_n,
        out=np.full(len(aggregate_conformal), np.nan),
        where=aggregate_conformal.accepted_n > 0,
    )
    delta = actions.delta.to_numpy()
    rng = np.random.default_rng(SEED + (1901 if context == "K562" else 1902))
    boot = np.array([
        rng.choice(delta, len(delta), replace=True).mean() for _ in range(10000)
    ])
    summary = {
        "context": context, "query_count": len(actions),
        "candidate_count": len(candidate_idx),
        "observed_utility": actions.actual_utility_observed.mean(),
        "reranked_utility": actions.actual_utility.mean(),
        "mean_delta": delta.mean(),
        "ci_low": np.quantile(boot, .025),
        "ci_high": np.quantile(boot, .975),
        "wilcoxon_p": float(wilcoxon(delta).pvalue),
        "positive_fold_count": int((folds_table.mean_delta > 0).sum()),
        "zero_mixing_fold_count": int((folds_table.selected_mixing == 0).sum()),
        "nonzero_risk_fold_count": int((folds_table.selected_risk > 0).sum()),
    }
    stratum_rows = []
    for stratum, group in actions.groupby("reliability_stratum"):
        values = group.delta.to_numpy()
        stratum_boot = np.array([
            rng.choice(values, len(values), replace=True).mean()
            for _ in range(5000)
        ])
        stratum_rows.append({
            "context": context, "reliability_stratum": stratum,
            "n": len(values), "observed_utility": group.actual_utility_observed.mean(),
            "reranked_utility": group.actual_utility.mean(),
            "mean_delta": values.mean(),
            "ci_low": np.quantile(stratum_boot, .025),
            "ci_high": np.quantile(stratum_boot, .975),
        })
    stratum_summary = pd.DataFrame(stratum_rows)
    reliable_row = stratum_summary[
        stratum_summary.reliability_stratum.eq("reliable_ge_0.30")
    ].iloc[0]
    primary = aggregate_conformal[
        aggregate_conformal.alpha.eq(.10)
        & aggregate_conformal.threshold_quantile.eq(.25)
    ].iloc[0]
    summary.update({
        "conformal_90_coverage": primary.empirical_bound_coverage,
        "q25_accepted_fraction": primary.accepted_fraction,
        "q25_accepted_success_fraction": primary.accepted_success_fraction,
        "reliable_stratum_n": int(reliable_row["n"]),
        "reliable_stratum_delta": reliable_row.mean_delta,
        "reliable_stratum_ci_low": reliable_row.ci_low,
        "reliable_stratum_ci_high": reliable_row.ci_high,
    })
    return (
        actions, folds_table, tuning, conformal,
        aggregate_conformal, stratum_summary, summary,
    )


def run(project_root: Path):
    out = project_root / "artifacts/nested_identity_cv"
    source = out / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    results = []
    for context, preparer in [
        ("K562", prepare_k562), ("RPE1", prepare_rpe1)
    ]:
        names, raw_frames = preparer(project_root)
        if context == "K562":
            frames = {
                split: raw_frames[split]["delta"]
                for split in ["train", "validation", "test"]
            }
        else:
            frames = raw_frames
        (
            actions, folds, tuning, conformal, aggregate_conformal,
            stratum_summary, summary,
        ) = run_context(
            names, frames, context
        )
        actions.to_csv(source / f"{context.lower()}_outer_test_actions.csv", index=False)
        folds.to_csv(source / f"{context.lower()}_fold_metrics.csv", index=False)
        tuning.to_csv(source / f"{context.lower()}_inner_tuning.csv", index=False)
        conformal.to_csv(source / f"{context.lower()}_conformal_metrics.csv", index=False)
        aggregate_conformal.to_csv(
            source / f"{context.lower()}_aggregate_conformal.csv", index=False
        )
        stratum_summary.to_csv(
            source / f"{context.lower()}_reliability_strata.csv", index=False
        )
        results.append(summary)
    summary = pd.DataFrame(results)
    summary.to_csv(source / "context_summary.csv", index=False)
    protocol = {
        "outer_validation": "five deterministic identity folds",
        "candidate_library": "deterministic train identities only",
        "query_pool": "all non-train identities; no outcome-dependent eligibility filter",
        "reliability_reporting": "train-validation cosine >=0.30 reported only as a prespecified stratum",
        "inner_roles": "60% fit, 20% tune, 20% conformal calibration by fold-specific hash",
        "model": "8 identity-bootstrap listwise rank-percentile regressors per outer fold",
        "shared_hyperparameter_grid": {
            "mixing": MIXING, "risk": RISKS,
        },
        "test_outcome_used_for_inner_selection": False,
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    lines = ["# Nested identity cross-validation across cell contexts", ""]
    for row in summary.itertuples(index=False):
        lines.extend([
            f"## {row.context}", "",
            f"- Queries/candidates: {row.query_count}/{row.candidate_count}.",
            f"- Observed versus re-ranked utility: {row.observed_utility:.3f} "
            f"versus {row.reranked_utility:.3f}.",
            f"- Mean delta: {row.mean_delta:.3f} "
            f"(95% identity-bootstrap CI {row.ci_low:.3f} to {row.ci_high:.3f}; "
            f"Wilcoxon P={row.wilcoxon_p:.3g}).",
            f"- Positive outer folds: {row.positive_fold_count}/5.",
            f"- Zero learned-score mixing: {row.zero_mixing_fold_count}/5 folds; "
            f"non-zero uncertainty risk: {row.nonzero_risk_fold_count}/5 folds.",
            f"- Aggregate alpha=0.10 conformal coverage: "
            f"{row.conformal_90_coverage:.3f}; 25th-percentile-threshold "
            f"acceptance: {row.q25_accepted_fraction:.3f}; accepted success: "
            f"{row.q25_accepted_success_fraction:.3f}.",
            f"- Prespecified reliable stratum (train–validation cosine ≥0.30; "
            f"n={row.reliable_stratum_n}): delta "
            f"{row.reliable_stratum_delta:.3f} "
            f"(95% CI {row.reliable_stratum_ci_low:.3f} to "
            f"{row.reliable_stratum_ci_high:.3f}).",
            "",
        ])
    (out / "NESTED_IDENTITY_CV_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n".join(lines))
