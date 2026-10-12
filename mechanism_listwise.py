from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .budgeted_equivalence import BUDGETS, EQUIVALENCE_FRACTIONS
from .distributional_ranking import distribution_frames, squared_distance
from .frangieh_a375 import CONDITIONS, V2_DISPERSION_WEIGHT, _control_z
from .k562_identity_holdout import identity_split
from .nested_identity_cv import outer_folds


BLENDS = (0.0, 0.25, 0.50, 0.75, 1.0)
SEED = 20260727


def _gene_symbols(project_root: Path, gene_ids: np.ndarray) -> np.ndarray:
    mapping = pd.read_csv(
        project_root / "artifacts/spectranav_v4_regulatory_prior/graphs/gene_mapping.csv"
    )
    lookup = dict(zip(mapping["gene_id"].astype(str), mapping["gene_symbol"].astype(str)))
    return np.asarray([lookup.get(str(g), "") for g in gene_ids])


def _row_normalized_matrix(
    modules: list[tuple[str, dict[str, float]]], symbols: np.ndarray
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    symbol_idx = {symbol: i for i, symbol in enumerate(symbols) if symbol}
    rows, cols, values, meta = [], [], [], []
    for name, weights in modules:
        mapped = [(symbol_idx[g], w) for g, w in weights.items() if g in symbol_idx]
        if not 5 <= len(mapped) <= 150:
            continue
        norm = math.sqrt(sum(w * w for _, w in mapped))
        row = len(meta)
        for col, weight in mapped:
            rows.append(row)
            cols.append(col)
            values.append(weight / max(norm, 1e-8))
        meta.append({"module": name, "mapped_gene_n": len(mapped)})
    matrix = sparse.csr_matrix(
        (values, (rows, cols)), shape=(len(meta), len(symbols)), dtype=np.float32
    )
    return matrix, pd.DataFrame(meta)


def build_module_matrices(project_root: Path, gene_ids: np.ndarray):
    symbols = _gene_symbols(project_root, gene_ids)
    archive = project_root / "data/cellnavigator/mechanism_networks/raw/ReactomePathways.gmt.zip"
    reactome = []
    with zipfile.ZipFile(archive) as zf:
        member = zf.namelist()[0]
        for line in zf.read(member).decode("utf-8").splitlines():
            fields = line.rstrip().split("\t")
            if len(fields) >= 4:
                reactome.append((fields[0], {g: 1.0 for g in fields[2:]}))
    reactome_matrix, reactome_meta = _row_normalized_matrix(reactome, symbols)

    edges = pd.read_csv(
        project_root / "data/cellnavigator/mechanism_networks/raw/dorothea_ABC_human.tsv",
        sep="\t",
    )
    confidence = {"A": 1.0, "B": 0.8, "C": 0.6}
    regulatory = []
    for tf, group in edges.groupby("source_genesymbol", observed=True):
        weights = {}
        for row in group.itertuples():
            sign = -1.0 if bool(row.is_inhibition) and not bool(row.is_stimulation) else 1.0
            weights[str(row.target_genesymbol)] = (
                sign * confidence.get(str(row.dorothea_level), 0.5)
            )
        regulatory.append((f"TF:{tf}", weights))
    regulatory_matrix, regulatory_meta = _row_normalized_matrix(regulatory, symbols)
    return {
        "reactome": (reactome_matrix, reactome_meta),
        "regulatory": (regulatory_matrix, regulatory_meta),
    }


def _project(frames: dict, matrix: sparse.csr_matrix) -> dict:
    return {
        split: {
            "names": frames[split]["names"],
            "mean": (matrix @ frames[split]["mean"].T).T.astype(np.float32),
            "dispersion": (matrix @ frames[split]["dispersion"].T).T.astype(np.float32),
        }
        for split in ("train", "validation", "test")
    }


def _pooled(frames: list[dict], split: str, component: str) -> np.ndarray:
    return np.mean([x[split][component] for x in frames], axis=0)


def _distance(query: dict, candidate_sources: list[dict], queries, candidates, qsplit, csplit):
    return squared_distance(
        query[qsplit]["mean"][queries],
        _pooled(candidate_sources, csplit, "mean")[candidates],
    ) + V2_DISPERSION_WEIGHT * squared_distance(
        query[qsplit]["dispersion"][queries],
        _pooled(candidate_sources, csplit, "dispersion")[candidates],
    )


def _rank_score(distance: np.ndarray) -> np.ndarray:
    order = np.argsort(distance, axis=1)
    inverse = np.empty_like(order)
    inverse[np.arange(len(order))[:, None], order] = np.arange(order.shape[1])
    return inverse.astype(np.float32) / max(order.shape[1] - 1, 1)


def _list_utility(score: np.ndarray, actual: np.ndarray, self_match: bool) -> float:
    score = score.copy()
    actual = actual.copy()
    if self_match:
        np.fill_diagonal(score, np.inf)
        np.fill_diagonal(actual, np.inf)
    order = np.argsort(score, axis=1)
    truth_order = np.argsort(actual, axis=1)
    truth_rank = np.empty_like(truth_order)
    truth_rank[np.arange(len(actual))[:, None], truth_order] = np.arange(actual.shape[1])
    n = actual.shape[1] - int(self_match)
    values = []
    for budget in (5, 10, 20):
        selected = order[:, :budget]
        best = np.min(np.take_along_axis(truth_rank, selected, axis=1), axis=1)
        for fraction in (0.01, 0.05, 0.10):
            values.append(float(np.mean(best < max(1, math.ceil(fraction * n)))))
    return float(np.mean(values))


def _select_blend(
    gene_score: np.ndarray, module_score: np.ndarray, actual: np.ndarray
) -> tuple[float, pd.DataFrame]:
    rows = []
    for blend in BLENDS:
        score = (1 - blend) * gene_score + blend * module_score
        rows.append({
            "blend": blend,
            "development_list_utility": _list_utility(score, actual, True),
        })
    table = pd.DataFrame(rows).sort_values(
        ["development_list_utility", "blend"], ascending=[False, True]
    )
    selected = float(table.iloc[0]["blend"])
    table["selected"] = table["blend"].eq(selected)
    return selected, table


def _evaluate(
    context: str, method: str, score: np.ndarray, actual: np.ndarray,
    names: np.ndarray, queries: np.ndarray, candidate_n: int,
) -> list[dict]:
    order = np.argsort(score, axis=1)
    truth_order = np.argsort(actual, axis=1)
    truth_rank = np.empty_like(truth_order)
    truth_rank[np.arange(len(actual))[:, None], truth_order] = np.arange(candidate_n)
    folds = outer_folds(names[queries])
    rows = []
    for qi, query in enumerate(names[queries]):
        for budget in BUDGETS:
            selected = order[qi, : min(budget, candidate_n)]
            best = selected[np.argmin(actual[qi, selected])]
            for fraction in EQUIVALENCE_FRACTIONS:
                relevant = max(1, math.ceil(fraction * candidate_n))
                rows.append({
                    "target_context": context, "outer_fold": int(folds[qi]),
                    "query": query, "method": method, "budget": budget,
                    "equivalence_fraction": fraction,
                    "equivalent_hit": float(
                        np.min(truth_rank[qi, selected]) < relevant
                    ),
                    "rank_utility": float(
                        1 - truth_rank[qi, best] / max(candidate_n - 1, 1)
                    ),
                    "best_actual_w2": float(actual[qi, best]),
                    "candidate_n": candidate_n,
                })
    return rows


def run(project_root: Path) -> None:
    cache = project_root / "artifacts/frangieh_a375_frozen_v1/cache"
    output = project_root / "artifacts/a375_mechanism_listwise_v1"
    source_dir = output / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    gene_ids = pd.read_csv(cache / "a375_distribution_genes.csv")["gene_id"].astype(str).to_numpy()
    matrices = build_module_matrices(project_root, gene_ids)
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(len(gene_ids))
    random_matrices = {
        name: matrix[:, permutation]
        for name, (matrix, _) in matrices.items()
    }
    module_manifest = []
    for name, (matrix, meta) in matrices.items():
        meta.assign(family=name).to_csv(source_dir / f"{name}_modules.csv", index=False)
        module_manifest.append({
            "family": name, "module_n": matrix.shape[0],
            "mapped_gene_n": int((np.asarray(matrix.getnnz(axis=0)) > 0).sum()),
        })
    pd.DataFrame(module_manifest).to_csv(source_dir / "module_manifest.csv", index=False)

    contexts = list(CONDITIONS.values())
    all_rows, tuning_rows = [], []
    for context in contexts:
        sources = [x for x in contexts if x != context]
        raw_target = distribution_frames(cache, context)
        z_target = _control_z(raw_target, cache, context)
        z_sources = [
            _control_z(distribution_frames(cache, source), cache, source)
            for source in sources
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
        gene_dev = _distance(
            z_target, z_sources, candidates, candidates, "train", "train"
        )
        gene_test = _distance(
            z_target, z_sources, queries, candidates, "validation", "train"
        )
        gene_dev_rank = _rank_score(gene_dev)
        gene_test_rank = _rank_score(gene_test)
        scores = {"Frozen v2": gene_test_rank}
        real_test_ranks, random_test_ranks = [], []
        for family, (matrix, _) in matrices.items():
            target_module = _project(z_target, matrix)
            source_module = [_project(x, matrix) for x in z_sources]
            dev = _rank_score(_distance(
                target_module, source_module, candidates, candidates, "train", "train"
            ))
            test = _rank_score(_distance(
                target_module, source_module, queries, candidates, "validation", "train"
            ))
            blend, table = _select_blend(gene_dev_rank, dev, actual_dev)
            table.insert(0, "family", family)
            table.insert(0, "target_context", context)
            tuning_rows.append(table)
            scores[f"{family.title()} gated"] = (1 - blend) * gene_test_rank + blend * test
            real_test_ranks.append(test)

            random_matrix = random_matrices[family]
            random_target = _project(z_target, random_matrix)
            random_sources = [_project(x, random_matrix) for x in z_sources]
            random_dev = _rank_score(_distance(
                random_target, random_sources, candidates, candidates, "train", "train"
            ))
            random_test = _rank_score(_distance(
                random_target, random_sources, queries, candidates, "validation", "train"
            ))
            random_blend, random_table = _select_blend(
                gene_dev_rank, random_dev, actual_dev
            )
            random_table.insert(0, "family", f"random_{family}")
            random_table.insert(0, "target_context", context)
            tuning_rows.append(random_table)
            random_test_ranks.append(random_test)

        combined_dev = np.mean([
            _rank_score(_distance(
                _project(z_target, matrix),
                [_project(x, matrix) for x in z_sources],
                candidates, candidates, "train", "train",
            ))
            for matrix, _ in matrices.values()
        ], axis=0)
        combined_test = np.mean(real_test_ranks, axis=0)
        blend, table = _select_blend(gene_dev_rank, combined_dev, actual_dev)
        table.insert(0, "family", "combined_mechanism")
        table.insert(0, "target_context", context)
        tuning_rows.append(table)
        scores["MechanismNav gated"] = (1 - blend) * gene_test_rank + blend * combined_test

        combined_random_dev = np.mean([
            _rank_score(_distance(
                _project(z_target, random_matrices[name]),
                [_project(x, random_matrices[name]) for x in z_sources],
                candidates, candidates, "train", "train",
            ))
            for name in matrices
        ], axis=0)
        combined_random_test = np.mean(random_test_ranks, axis=0)
        random_blend, table = _select_blend(
            gene_dev_rank, combined_random_dev, actual_dev
        )
        table.insert(0, "family", "combined_random")
        table.insert(0, "target_context", context)
        tuning_rows.append(table)
        scores["Random-module gated"] = (
            (1 - random_blend) * gene_test_rank
            + random_blend * combined_random_test
        )
        for method, score in scores.items():
            all_rows.extend(_evaluate(
                context, method, score, actual_test, names, queries, len(candidates)
            ))

    result = pd.DataFrame(all_rows)
    result.to_csv(source_dir / "per_query_budget_metrics.csv", index=False)
    pd.concat(tuning_rows, ignore_index=True).to_csv(
        source_dir / "development_listwise_tuning.csv", index=False
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
        "real_modules": ["Reactome", "DoRothEA A-C signed confidence-weighted regulons"],
        "random_control": "single fixed gene-column permutation preserving module sizes and weights",
        "optimization": "direct listwise budget utility on candidate train/validation guides",
        "blend_grid": BLENDS,
        "zero_calibration_rejection": True,
    }, indent=2), encoding="utf-8")
