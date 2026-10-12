from __future__ import annotations

import gzip
import json
import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.utils import sparsefuncs


SEED = 20260723
METHODS = ["Random", "Mean", "Additive", "Ridge", "Observed-NN"]
COLORS = {
    "Random": "#C9CDD4",
    "Mean": "#AEBBD2",
    "Additive": "#83A7C9",
    "Ridge": "#527FAF",
    "Observed-NN": "#2F5D8A",
}


@dataclass(frozen=True)
class NormanPaths:
    matrix: Path
    barcodes: Path
    genes: Path
    identities: Path


def canonicalize_guide(value: str) -> str:
    """Canonical perturbation label invariant to guide order and controls."""
    raw = str(value).split("__", 1)[0]
    genes = sorted({x for x in raw.split("_") if x and not x.startswith("NegCtrl")})
    return "+".join(genes) if genes else "CTRL"


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    an = np.linalg.norm(a, axis=1, keepdims=True)
    bn = np.linalg.norm(b, axis=1, keepdims=True)
    an[an == 0] = 1
    bn[bn == 0] = 1
    return (a / an) @ (b / bn).T


def row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(ac, axis=1) * np.linalg.norm(bc, axis=1)
    out = np.divide((ac * bc).sum(axis=1), denom, out=np.zeros(len(a)), where=denom > 0)
    return out


def inverse_ranks(similarity: np.ndarray, queries: list[str], candidates: list[str]) -> np.ndarray:
    lookup = {name: i for i, name in enumerate(candidates)}
    ranks = np.empty(len(queries), dtype=int)
    for i, q in enumerate(queries):
        order = np.argsort(-similarity[i], kind="stable")
        ranks[i] = int(np.flatnonzero(order == lookup[q])[0]) + 1
    return ranks


def systema_centroid_accuracy(pred_delta: np.ndarray, true_delta: np.ndarray) -> float:
    """Systema-style centroid discrimination on perturbation-specific deltas."""
    sim = cosine_matrix(pred_delta, true_delta)
    return float(np.mean(np.argmax(sim, axis=1) == np.arange(len(true_delta))))


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 1000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _read_genes(path: Path) -> pd.DataFrame:
    genes = pd.read_csv(path, sep="\t", header=None)
    if genes.shape[1] >= 2:
        genes = genes.iloc[:, :2]
        genes.columns = ["gene_id", "gene_symbol"]
    else:
        genes.columns = ["gene_symbol"]
        genes["gene_id"] = genes["gene_symbol"]
    return genes


def _load_and_normalize(paths: NormanPaths, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_npz = cache_dir / "norman_log1p_cells_by_genes.npz"
    cache_obs = cache_dir / "norman_cell_metadata.parquet"
    cache_genes = cache_dir / "norman_genes.parquet"
    if cache_npz.exists() and cache_obs.exists() and cache_genes.exists():
        return sparse.load_npz(cache_npz), pd.read_parquet(cache_obs), pd.read_parquet(cache_genes)

    barcodes = pd.read_csv(paths.barcodes, sep="\t", header=None, names=["cell_barcode"])
    obs = pd.read_csv(paths.identities)
    obs["perturbation"] = obs["guide_identity"].map(canonicalize_guide)
    obs["split"] = obs["gemgroup"].map(
        lambda x: "train" if x in {1, 2, 3, 4, 5} else ("validation" if x == 6 else "test")
    )
    obs = barcodes.merge(obs, on="cell_barcode", how="left", sort=False)
    keep = obs["guide_identity"].notna().to_numpy()
    obs = obs.loc[keep].reset_index(drop=True)

    with gzip.open(paths.matrix, "rb") as handle:
        matrix = mmread(handle)
    matrix = matrix.tocsr().T.tocsr()[keep].astype(np.float32)
    matrix.sum_duplicates()
    library = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(1e4, library, out=np.zeros_like(library, dtype=np.float32), where=library > 0)
    sparsefuncs.inplace_csr_row_scale(matrix, scale)
    np.log1p(matrix.data, out=matrix.data)
    matrix.eliminate_zeros()

    genes = _read_genes(paths.genes)
    sparse.save_npz(cache_npz, matrix, compressed=True)
    obs.to_parquet(cache_obs, index=False)
    genes.to_parquet(cache_genes, index=False)
    return matrix, obs, genes


def _aggregate_centroids(matrix, obs: pd.DataFrame):
    group_key = obs["split"].astype(str) + "|" + obs["perturbation"].astype(str)
    categories, group_index = np.unique(group_key, return_inverse=True)
    membership = sparse.csr_matrix(
        (np.ones(len(obs), dtype=np.float32), (group_index, np.arange(len(obs)))),
        shape=(len(categories), len(obs)),
    )
    sums = membership @ matrix
    counts = np.bincount(group_index).astype(np.float32)
    sparsefuncs.inplace_csr_row_scale(sums, 1.0 / counts)
    centroids = sums.toarray().astype(np.float32)
    meta = pd.DataFrame({"group": categories, "n_cells": counts.astype(int)})
    meta[["split", "perturbation"]] = meta["group"].str.split("|", n=1, expand=True)
    return centroids, meta


def _make_delta_tables(centroids: np.ndarray, meta: pd.DataFrame, n_features: int = 2000):
    frames = {}
    common = None
    for split in ["train", "validation", "test"]:
        part = meta[meta["split"] == split].copy()
        idx = part.index.to_numpy()
        names = part["perturbation"].tolist()
        ctrl = centroids[idx[names.index("CTRL")]]
        mask = part["perturbation"].ne("CTRL").to_numpy()
        split_names = part.loc[mask, "perturbation"].tolist()
        split_delta = centroids[idx[mask]] - ctrl
        split_post = centroids[idx[mask]]
        split_counts = part.loc[mask, "n_cells"].to_numpy()
        frames[split] = {
            "names": split_names,
            "delta": split_delta,
            "post": split_post,
            "control": ctrl,
            "counts": split_counts,
        }
        common = set(split_names) if common is None else common.intersection(split_names)
    names = sorted(common)
    for split in frames:
        order = {x: i for i, x in enumerate(frames[split]["names"])}
        ids = np.array([order[x] for x in names])
        for key in ["delta", "post", "counts"]:
            frames[split][key] = frames[split][key][ids]
        frames[split]["names"] = names

    train_delta = frames["train"]["delta"]
    variance = train_delta.var(axis=0)
    keep = np.argsort(variance)[-min(n_features, len(variance)) :]
    keep.sort()
    for split in frames:
        frames[split]["delta"] = frames[split]["delta"][:, keep]
        frames[split]["post"] = frames[split]["post"][:, keep]
        frames[split]["control"] = frames[split]["control"][keep]
    return frames, keep


def _perturbation_design(names: list[str]):
    all_genes = sorted({g for name in names for g in name.split("+")})
    col = {g: i for i, g in enumerate(all_genes)}
    design = np.zeros((len(names), len(all_genes)), dtype=np.float32)
    for i, name in enumerate(names):
        for gene in name.split("+"):
            design[i, col[gene]] = 1
    return design, all_genes


def _build_predictions(frames: dict, rng: np.random.Generator):
    names = frames["train"]["names"]
    train = frames["train"]["delta"]
    design, genes = _perturbation_design(names)

    single = {}
    for i, name in enumerate(names):
        if "+" not in name:
            single[name] = train[i]
    global_mean = train.mean(axis=0)
    additive = np.stack(
        [sum((single.get(g, global_mean * 0.5) for g in name.split("+")), start=np.zeros_like(global_mean)) for name in names]
    )
    ridge = Ridge(alpha=10.0, fit_intercept=True)
    ridge.fit(design, train)
    ridge_pred = ridge.predict(design).astype(np.float32)
    random_pred = train[rng.permutation(len(train))]
    mean_pred = np.repeat(global_mean[None, :], len(train), axis=0)
    return {
        "Random": random_pred,
        "Mean": mean_pred,
        "Additive": additive.astype(np.float32),
        "Ridge": ridge_pred,
        "Observed-NN": train,
    }, genes


def _evaluate(frames: dict, predictions: dict, rng: np.random.Generator):
    names = frames["test"]["names"]
    truth = frames["test"]["delta"]
    # Select candidates from train-derived model predictions, then replay the
    # recommendation against independent test-gem-group observations.
    observed_candidates = frames["test"]["delta"]
    forward_rows = []
    closed_rows = []
    open_rows = []

    actual_open_similarity = cosine_matrix(truth, observed_candidates)
    for i in range(len(names)):
        actual_open_similarity[i, i] = -np.inf
    finite_open = np.where(np.isfinite(actual_open_similarity), actual_open_similarity, np.nan)
    thresholds = np.nanquantile(finite_open, 0.90, axis=1)
    best_open = np.nanmax(finite_open, axis=1)

    for method, pred in predictions.items():
        pcc = row_pearson(pred, truth)
        cos = np.diag(cosine_matrix(pred, truth))
        centroid_acc = systema_centroid_accuracy(pred, truth)
        for i, name in enumerate(names):
            forward_rows.append(
                {
                    "method": method,
                    "perturbation": name,
                    "delta_pearson": pcc[i],
                    "delta_cosine": cos[i],
                    "centroid_correct": int(np.argmax(cosine_matrix(pred[i : i + 1], truth)[0]) == i),
                    "centroid_accuracy_overall": centroid_acc,
                }
            )

        if method == "Random":
            scores = rng.random((len(names), len(names)))
        else:
            scores = cosine_matrix(truth, pred)
        ranks = inverse_ranks(scores, names, names)
        for name, rank in zip(names, ranks):
            closed_rows.append(
                {
                    "method": method,
                    "perturbation": name,
                    "rank": int(rank),
                    "recall_at_1": int(rank <= 1),
                    "recall_at_5": int(rank <= 5),
                    "recall_at_10": int(rank <= 10),
                    "reciprocal_rank": 1.0 / rank,
                    "ndcg": 1.0 / math.log2(rank + 1),
                }
            )

        open_scores = scores.copy()
        np.fill_diagonal(open_scores, -np.inf)
        chosen = np.argmax(open_scores, axis=1)
        chosen_actual = actual_open_similarity[np.arange(len(names)), chosen]
        for i, name in enumerate(names):
            open_rows.append(
                {
                    "method": method,
                    "perturbation": name,
                    "recommended": names[chosen[i]],
                    "actual_target_cosine": chosen_actual[i],
                    "best_available_cosine": best_open[i],
                    "regret": best_open[i] - chosen_actual[i],
                    "functional_hit_top10pct": int(chosen_actual[i] >= thresholds[i]),
                }
            )

    return pd.DataFrame(forward_rows), pd.DataFrame(closed_rows), pd.DataFrame(open_rows)


def _summarize(forward: pd.DataFrame, closed: pd.DataFrame, open_df: pd.DataFrame):
    rng = np.random.default_rng(SEED + 1)
    rows = []
    metric_specs = [
        ("forward_delta_pearson", forward, "delta_pearson"),
        ("forward_centroid_accuracy", forward, "centroid_correct"),
        ("closed_mrr", closed, "reciprocal_rank"),
        ("closed_recall_at_5", closed, "recall_at_5"),
        ("open_target_cosine", open_df, "actual_target_cosine"),
        ("open_functional_hit", open_df, "functional_hit_top10pct"),
        ("open_regret", open_df, "regret"),
    ]
    for metric, frame, column in metric_specs:
        for method in METHODS:
            vals = frame.loc[frame["method"] == method, column].to_numpy(dtype=float)
            mean, lo, hi = _bootstrap_mean_ci(vals, rng)
            rows.append({"metric": metric, "method": method, "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)})
    return pd.DataFrame(rows)


def _panel_label(ax, label: str):
    ax.text(-0.10, 1.05, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def _save_figure(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")


def _box(ax, xy, width, height, text, fc, ec="#58616C", fontsize=5.8):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.8,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)


def plot_figure1(meta: pd.DataFrame, frames: dict, outdir: Path):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
        }
    )
    fig = plt.figure(figsize=(7.2, 6.1))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.36, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    _panel_label(ax_a, "a")
    stages = [
        (0.01, "Norman\nPerturb-seq\n111,445 cells", "#E7EEF6"),
        (0.22, "Gem-group split\ntrain 1–5 | val 6 | test 7–8", "#DCE8F3"),
        (0.41, "Target queries\nclosed | open\nunsupported", "#C9DDEC"),
        (0.62, "Candidate models\nsimple → learned baselines", "#AFCBE0"),
        (0.81, "Decision scores\nretrieval | replay\ncalibration", "#91B6D3"),
    ]
    for x, text, fc in stages:
        _box(ax_a, (x, 0.55), 0.16, 0.25, text, fc)
    for x in [0.18, 0.38, 0.58, 0.78]:
        ax_a.add_patch(FancyArrowPatch((x, 0.675), (x + 0.035, 0.675), arrowstyle="-|>", mutation_scale=10, lw=0.9))

    tasks = [
        (0.12, "Closed set", "Recover the exact\nheld-out perturbation"),
        (0.37, "Open set", "Remove the true action;\nfind a functional analogue"),
        (0.62, "Reachability", "Estimate support and\nabstain when needed"),
        (0.87, "Experiment replay", "Select hidden experiments\nunder a fixed budget"),
    ]
    for x, title, text in tasks:
        ax_a.text(x, 0.34, title, ha="center", va="center", fontweight="bold", color="#2F5D8A")
        ax_a.text(x, 0.18, text, ha="center", va="center", fontsize=6.2, color="#3F4650")
    ax_a.text(0.5, 0.94, "CellPilot-Eval converts a static screen into goal-directed decision tasks", ha="center", fontsize=9, fontweight="bold")

    _panel_label(ax_b, "b")
    split_counts = meta.groupby("split")["n_cells"].sum().reindex(["train", "validation", "test"])
    ax_b.bar(split_counts.index, split_counts.values / 1000, color=["#527FAF", "#91B6D3", "#C9DDEC"])
    ax_b.set_ylabel("Annotated cells (×10³)")
    ax_b.set_title("Leakage-resistant gem-group split", fontsize=7.5)
    ax_b.tick_params(axis="x", rotation=25)

    _panel_label(ax_c, "c")
    names = frames["train"]["names"]
    n_genes = np.array([1 if "+" not in x else 2 for x in names])
    vals = [int((n_genes == 1).sum()), int((n_genes == 2).sum())]
    ax_c.bar(["single", "double"], vals, color=["#83A7C9", "#2F5D8A"])
    ax_c.set_ylabel("Canonical perturbations")
    ax_c.set_title("Candidate action space", fontsize=7.5)
    for i, v in enumerate(vals):
        ax_c.text(i, v + max(vals) * 0.03, str(v), ha="center", fontsize=6.5)

    _panel_label(ax_d, "d")
    layers = ["Molecular", "Functional", "Decision"]
    metrics = ["Δ-correlation\ncentroid accuracy", "target cosine\npathway equivalence", "MRR / Recall@k\nrisk–coverage"]
    y = np.arange(3)
    ax_d.barh(y, [1, 1, 1], color=["#DCE8F3", "#AFCBE0", "#527FAF"])
    ax_d.set_yticks(y, layers)
    ax_d.set_xticks([])
    ax_d.invert_yaxis()
    for yi, text in zip(y, metrics):
        ax_d.text(0.5, yi, text, ha="center", va="center", fontsize=6.2, color="#1E2933")
    ax_d.set_title("Three-level evidence ladder", fontsize=7.5)
    for spine in ax_d.spines.values():
        spine.set_visible(False)

    fig.suptitle("Norman MVP: from public perturbation data to inverse virtual-cell evaluation", y=0.995, fontsize=10, fontweight="bold")
    _save_figure(fig, outdir / "Figure1_Norman_MVP_framework")
    plt.close(fig)


def plot_figure2(summary: pd.DataFrame, forward: pd.DataFrame, closed: pd.DataFrame, open_df: pd.DataFrame, outdir: Path):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        }
    )
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.38)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes

    def bars(ax, metric, title, ylabel):
        d = summary[summary["metric"] == metric].set_index("method").loc[METHODS]
        x = np.arange(len(METHODS))
        ax.bar(x, d["mean"], color=[COLORS[m] for m in METHODS], edgecolor="white", linewidth=0.4)
        ax.errorbar(
            x,
            d["mean"],
            yerr=np.vstack([d["mean"] - d["ci_low"], d["ci_high"] - d["mean"]]),
            fmt="none",
            ecolor="#26323D",
            elinewidth=0.7,
            capsize=2,
        )
        ax.set_xticks(x, METHODS, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=7.5)

    bars(ax_a, "forward_delta_pearson", "Forward perturbation-specific fit", "Δ Pearson")
    bars(ax_b, "closed_mrr", "Closed-set inverse retrieval", "Mean reciprocal rank")
    bars(ax_c, "open_target_cosine", "Open-set target attainment", "Observed target cosine")
    bars(ax_d, "open_functional_hit", "Functional analogue recovery", "Top-decile hit rate")

    forward_rank = (
        forward.groupby("method")["delta_pearson"].mean().rank(ascending=False, method="min").reindex(METHODS)
    )
    inverse_rank = (
        closed.groupby("method")["reciprocal_rank"].mean().rank(ascending=False, method="min").reindex(METHODS)
    )
    _panel_label(ax_e, "e")
    for method in METHODS:
        ax_e.scatter(forward_rank[method], inverse_rank[method], s=42, color=COLORS[method], edgecolor="white", linewidth=0.6)
        ax_e.text(forward_rank[method] + 0.07, inverse_rank[method] + 0.07, method, fontsize=5.7)
    ax_e.plot([1, len(METHODS)], [1, len(METHODS)], "--", color="#A7ADB5", lw=0.8)
    ax_e.set_xlim(0.7, len(METHODS) + 0.6)
    ax_e.set_ylim(len(METHODS) + 0.6, 0.7)
    ax_e.set_xlabel("Forward rank")
    ax_e.set_ylabel("Closed inverse rank")
    rho = spearmanr(forward_rank, inverse_rank).statistic
    ax_e.set_title(f"Forward–inverse concordance (Spearman ρ={rho:.2f})", fontsize=7.5)

    _panel_label(ax_f, "f")
    closed_hit = closed.groupby("method")["recall_at_5"].mean().reindex(METHODS)
    open_hit = open_df.groupby("method")["functional_hit_top10pct"].mean().reindex(METHODS)
    y = np.arange(len(METHODS))
    ax_f.hlines(y, open_hit, closed_hit, color="#B6BDC5", lw=1.4)
    ax_f.scatter(closed_hit, y, color="#2F5D8A", s=28, label="Closed Recall@5", zorder=3)
    ax_f.scatter(open_hit, y, color="#C27A68", s=28, label="Open functional hit", zorder=3)
    ax_f.set_yticks(y, METHODS)
    ax_f.invert_yaxis()
    ax_f.set_xlabel("Success rate")
    ax_f.set_xlim(-0.02, 1.02)
    ax_f.set_title("Closed-to-open performance gap", fontsize=7.5)
    ax_f.legend(fontsize=5.7, loc="lower right")

    for label, ax in zip("abcd", [ax_a, ax_b, ax_c, ax_d]):
        _panel_label(ax, label)
    fig.suptitle("Closed and open tests reveal distinct intervention capabilities", y=0.995, fontsize=10, fontweight="bold")
    _save_figure(fig, outdir / "Figure2_closed_open_results")
    plt.close(fig)


def _write_report(
    outdir: Path,
    frames: dict,
    summary: pd.DataFrame,
    forward: pd.DataFrame,
    closed: pd.DataFrame,
    open_df: pd.DataFrame,
):
    table = summary.pivot(index="method", columns="metric", values="mean").reindex(METHODS)
    rho = spearmanr(
        forward.groupby("method")["delta_pearson"].mean().reindex(METHODS),
        closed.groupby("method")["reciprocal_rank"].mean().reindex(METHODS),
    ).statistic
    best_closed = table["closed_mrr"].idxmax()
    best_open = table["open_target_cosine"].idxmax()
    report = f"""# Norman MVP results

## Scope

- Dataset: Norman combinatorial Perturb-seq.
- Split: gem groups 1–5 train, 6 validation, 7–8 test.
- Queries/candidates: {len(frames['train']['names'])} canonical non-control perturbations.
- Features: {frames['train']['delta'].shape[1]} genes selected by training-delta variance.
- Baselines: {", ".join(METHODS)}.

## Main observations

- Best closed-set MRR: **{best_closed}** ({table.loc[best_closed, 'closed_mrr']:.3f}).
- Best open-set observed target cosine: **{best_open}** ({table.loc[best_open, 'open_target_cosine']:.3f}).
- Spearman association between mean forward delta correlation and closed inverse MRR: **{rho:.3f}**.
- Open-set removes the original perturbation and evaluates the observed effect of the recommended alternative.

## Interpretation boundary

This MVP evaluates empirical target support inside the Norman candidate library. It does not establish therapeutic efficacy, universal biological reachability, or true sequential intervention control.

## Files

- `source_data/forward_per_query.csv`
- `source_data/closed_per_query.csv`
- `source_data/open_per_query.csv`
- `source_data/summary_metrics.csv`
- `figures/Figure1_Norman_MVP_framework.*`
- `figures/Figure2_closed_open_results.*`
"""
    (outdir / "Norman_MVP_report.md").write_text(report, encoding="utf-8")


def run_norman_mvp(project_root: Path):
    data_dir = project_root / "data" / "cellnavigator" / "genetic" / "norman"
    outdir = project_root / "artifacts" / "norman_mvp"
    source = outdir / "source_data"
    figures = outdir / "figures"
    cache = outdir / "cache"
    source.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    paths = NormanPaths(
        matrix=data_dir / "GSE133344_filtered_matrix.mtx.gz",
        barcodes=data_dir / "GSE133344_filtered_barcodes.tsv.gz",
        genes=data_dir / "GSE133344_filtered_genes.tsv.gz",
        identities=data_dir / "GSE133344_filtered_cell_identities.csv.gz",
    )
    matrix, obs, genes = _load_and_normalize(paths, cache)
    centroids, meta = _aggregate_centroids(matrix, obs)
    frames, selected = _make_delta_tables(centroids, meta, n_features=2000)
    selected_genes = genes.iloc[selected].copy()
    selected_genes["feature_index"] = selected

    rng = np.random.default_rng(SEED)
    predictions, perturbation_genes = _build_predictions(frames, rng)
    forward, closed, open_df = _evaluate(frames, predictions, rng)
    summary = _summarize(forward, closed, open_df)

    obs_qc = (
        obs.groupby(["split", "perturbation"], as_index=False)
        .size()
        .rename(columns={"size": "n_cells"})
    )
    obs_qc["n_genes_perturbed"] = obs_qc["perturbation"].map(lambda x: 0 if x == "CTRL" else x.count("+") + 1)
    obs_qc.to_csv(source / "norman_condition_counts.csv", index=False)
    meta.to_csv(source / "centroid_groups.csv", index=False)
    selected_genes.to_csv(source / "selected_genes.csv", index=False)
    pd.DataFrame({"perturbation": frames["train"]["names"]}).to_csv(source / "candidate_library.csv", index=False)
    forward.to_csv(source / "forward_per_query.csv", index=False)
    closed.to_csv(source / "closed_per_query.csv", index=False)
    open_df.to_csv(source / "open_per_query.csv", index=False)
    summary.to_csv(source / "summary_metrics.csv", index=False)
    np.savez_compressed(
        source / "norman_centroid_deltas.npz",
        train=frames["train"]["delta"],
        validation=frames["validation"]["delta"],
        test=frames["test"]["delta"],
        perturbations=np.array(frames["train"]["names"]),
        feature_indices=selected,
    )
    split_manifest = {
        "seed": SEED,
        "train_gemgroups": [1, 2, 3, 4, 5],
        "validation_gemgroups": [6],
        "test_gemgroups": [7, 8],
        "candidate_count": len(frames["train"]["names"]),
        "feature_count": len(selected),
        "normalization": "library-size 1e4 followed by log1p; group mean by canonical perturbation",
    }
    (source / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    plot_figure1(meta, frames, figures)
    plot_figure2(summary, forward, closed, open_df, figures)
    _write_report(outdir, frames, summary, forward, closed, open_df)
    print(summary.to_string(index=False))
