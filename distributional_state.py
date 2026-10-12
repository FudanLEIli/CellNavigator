from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .k562_mvp import split_from_gem
from .rpe1_context_transfer import split_rpe1


def aggregate_gaussian_summaries(
    h5ad_path: Path, centroid_meta_path: Path, feature_path: Path,
    output_dir: Path, context: str, chunk_size: int = 4000,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{context.lower()}_gaussian_summaries.npz"
    meta_out = output_dir / f"{context.lower()}_distribution_metadata.csv"
    genes_out = output_dir / f"{context.lower()}_distribution_genes.csv"
    if all(path.exists() for path in [npz_path, meta_out, genes_out]):
        return
    group_meta = pd.read_csv(centroid_meta_path)
    feature_genes = pd.read_csv(feature_path).gene_id.astype(str).tolist()
    a = ad.read_h5ad(h5ad_path, backed="r")
    gene_lookup = pd.Series(
        np.arange(a.n_vars), index=a.var_names.astype(str)
    )
    shared = [gene for gene in feature_genes if gene in gene_lookup.index]
    selected = np.sort(gene_lookup.loc[shared].to_numpy())
    selected_genes = a.var_names[selected].astype(str).to_numpy()
    obs = a.obs[["gene", "gem_group"]].copy()
    obs["gene"] = obs.gene.astype(str)
    obs["split"] = (
        split_from_gem(obs.gem_group) if context == "K562"
        else split_rpe1(obs.gem_group)
    ).astype(str)
    group_index = {
        (str(row.split), str(row.perturbation)): i
        for i, row in group_meta.iterrows()
    }
    row_groups = np.asarray([
        group_index.get((split, gene), -1)
        for split, gene in zip(obs.split, obs.gene)
    ], dtype=np.int32)
    sums = np.zeros((len(group_meta), len(selected)), dtype=np.float64)
    sums_sq = np.zeros_like(sums)
    counts = np.zeros(len(group_meta), dtype=np.int64)
    for start in range(0, a.n_obs, chunk_size):
        end = min(start + chunk_size, a.n_obs)
        groups = row_groups[start:end]
        keep = groups >= 0
        if not keep.any():
            continue
        block = a.X[start:end, selected]
        if sparse.issparse(block):
            block = block.toarray()
        block = np.asarray(block, dtype=np.float32)[keep]
        active = groups[keep]
        unique, inverse = np.unique(active, return_inverse=True)
        indicator = sparse.csr_matrix(
            (np.ones(len(inverse)), (inverse, np.arange(len(inverse)))),
            shape=(len(unique), len(inverse)),
        )
        sums[unique] += indicator @ block
        sums_sq[unique] += indicator @ np.square(block)
        counts[unique] += np.bincount(inverse, minlength=len(unique))
        if start % 40000 == 0:
            print(
                f"{context} distribution aggregation: {end:,}/{a.n_obs:,}",
                flush=True,
            )
    a.file.close()
    means = sums / np.maximum(counts[:, None], 1)
    variances = sums_sq / np.maximum(counts[:, None], 1) - means ** 2
    variances = np.maximum(variances, 1e-6)
    np.savez_compressed(
        npz_path, means=means.astype(np.float32),
        variances=variances.astype(np.float32), counts=counts,
    )
    result_meta = group_meta.copy()
    result_meta["distribution_n_cells"] = counts
    result_meta.to_csv(meta_out, index=False)
    pd.DataFrame({"gene_id": selected_genes}).to_csv(genes_out, index=False)
    audit = {
        "context": context, "cell_count": int(counts.sum()),
        "group_count": len(group_meta), "feature_count": len(selected),
        "minimum_group_cells": int(counts.min()),
        "summary": "diagonal Gaussian mean and population variance",
    }
    (output_dir / f"{context.lower()}_aggregation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


def run(project_root: Path):
    out = project_root / "artifacts/distributional_state/cache"
    aggregate_gaussian_summaries(
        project_root / "data/cellnavigator/genetic/replogle_k562"
        / "K562_essential_normalized_singlecell_01.h5ad",
        project_root / "artifacts/k562_mvp/cache/k562_centroid_metadata.csv",
        project_root / "artifacts/k562_mvp/source_data/selected_genes.csv",
        out, "K562",
    )
    aggregate_gaussian_summaries(
        project_root / "data/cellnavigator/genetic/replogle_rpe1"
        / "rpe1_normalized_singlecell_01.h5ad",
        project_root / "artifacts/rpe1_context_transfer/cache"
        / "rpe1_centroid_metadata.csv",
        project_root / "artifacts/rpe1_context_transfer/source_data"
        / "aligned_features.csv",
        out, "RPE1",
    )
