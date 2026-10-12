from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wilcoxon

from .norman_mvp import cosine_matrix, inverse_ranks, row_pearson


SEED = 20260724
SPLITS = ["train", "validation", "test"]
METHODS = ["Mean", "K562 transfer", "RPE1 in-context"]
COLORS = {"Mean": "#C8CCD2", "K562 transfer": "#D48B6A", "RPE1 in-context": "#255B78"}
GRID, INK = "#E5E8EA", "#22282E"


def split_rpe1(gem: pd.Series) -> pd.Categorical:
    return pd.cut(gem.astype(int), bins=[0, 36, 46, 56], labels=SPLITS, include_lowest=True)


def aggregate_rpe1(h5ad_path: Path, cache: Path, min_cells: int = 10):
    cache.mkdir(parents=True, exist_ok=True)
    npz_path = cache / "rpe1_centroids_all_genes.npz"
    meta_path = cache / "rpe1_centroid_metadata.csv"
    genes_path = cache / "rpe1_genes.csv"
    if all(p.exists() for p in [npz_path, meta_path, genes_path]):
        z = np.load(npz_path)
        return z["centroids"], pd.read_csv(meta_path), pd.read_csv(genes_path)

    a = ad.read_h5ad(h5ad_path, backed="r")
    obs = a.obs[["gene", "gem_group"]].copy()
    obs["gene"] = obs["gene"].astype(str)
    obs["split"] = split_rpe1(obs["gem_group"])
    counts = obs.groupby(["gene", "split"], observed=True).size().unstack(fill_value=0).reindex(columns=SPLITS)
    eligible = counts.index[counts.min(axis=1) >= min_cells].astype(str).tolist()
    if "non-targeting" not in eligible:
        raise ValueError("RPE1 control failed the minimum-cell rule")
    perturbations = ["non-targeting"] + sorted(x for x in eligible if x != "non-targeting")
    group_names = [(split, gene) for split in SPLITS for gene in perturbations]
    group_index = {key: i for i, key in enumerate(group_names)}
    row_groups = np.array([
        group_index.get((str(split), gene), -1)
        for split, gene in zip(obs["split"].astype(str), obs["gene"])
    ], dtype=np.int32)
    sums = np.zeros((len(group_names), a.n_vars), dtype=np.float64)
    ns = np.zeros(len(group_names), dtype=np.int64)
    for start in range(0, a.n_obs, 5000):
        end = min(start + 5000, a.n_obs)
        groups = row_groups[start:end]
        keep = groups >= 0
        if not keep.any():
            continue
        x = np.asarray(a.X[start:end, :], dtype=np.float32)[keep]
        active = groups[keep]
        unique, inverse = np.unique(active, return_inverse=True)
        indicator = sparse.csr_matrix(
            (np.ones(len(inverse)), (inverse, np.arange(len(inverse)))),
            shape=(len(unique), len(inverse)),
        )
        sums[unique] += indicator @ x
        ns[unique] += np.bincount(inverse, minlength=len(unique))
        if start % 50000 == 0:
            print(f"RPE1 aggregation: {end:,}/{a.n_obs:,} cells", flush=True)
    var_names = a.var_names.astype(str).to_numpy()
    gene_names = a.var["gene_name"].astype(str).to_numpy()
    a.file.close()
    centroids = (sums / ns[:, None]).astype(np.float32)
    meta = pd.DataFrame(group_names, columns=["split", "perturbation"])
    meta["n_cells"] = ns
    genes = pd.DataFrame({"gene_id": var_names, "gene_name": gene_names})
    np.savez_compressed(npz_path, centroids=centroids)
    meta.to_csv(meta_path, index=False)
    genes.to_csv(genes_path, index=False)
    return centroids, meta, genes


def all_gene_deltas(centroids, meta):
    out = {}
    for split in SPLITS:
        idx = meta.index[meta.split == split].to_numpy()
        names = meta.loc[idx, "perturbation"].astype(str).to_numpy()
        values = centroids[idx]
        ctrl = values[np.flatnonzero(names == "non-targeting")[0]]
        keep = names != "non-targeting"
        out[split] = {"names": names[keep], "delta": (values[keep] - ctrl).astype(np.float32)}
    return out


def align_contexts(k_frames, k_genes, r_frames, r_genes, n_features=2000):
    k_gene = pd.Series(np.arange(len(k_genes)), index=k_genes.gene_id.astype(str))
    r_gene = pd.Series(np.arange(len(r_genes)), index=r_genes.gene_id.astype(str))
    shared_genes = sorted(set(k_gene.index) & set(r_gene.index))
    ki = k_gene.loc[shared_genes].to_numpy()
    ri = r_gene.loc[shared_genes].to_numpy()
    shared_actions = sorted(set(k_frames["train"]["names"]) & set(r_frames["train"]["names"]))
    k_action = {n: i for i, n in enumerate(k_frames["train"]["names"])}
    r_action = {n: i for i, n in enumerate(r_frames["train"]["names"])}
    ka = np.array([k_action[n] for n in shared_actions])
    ra = np.array([r_action[n] for n in shared_actions])
    k_train = k_frames["train"]["delta"][ka][:, ki]
    r_train = r_frames["train"]["delta"][ra][:, ri]
    pooled_variance = k_train.var(axis=0) + r_train.var(axis=0)
    selected_local = np.argsort(pooled_variance)[-min(n_features, len(shared_genes)):]
    scale = np.sqrt((k_train[:, selected_local].var(axis=0) + r_train[:, selected_local].var(axis=0)) / 2) + 1e-6
    aligned = {"names": np.asarray(shared_actions), "genes": np.asarray(shared_genes)[selected_local]}
    for context, frames, action_idx, gene_idx in [
        ("k562", k_frames, ka, ki), ("rpe1", r_frames, ra, ri)
    ]:
        aligned[context] = {}
        for split in SPLITS:
            name_map = {n: i for i, n in enumerate(frames[split]["names"])}
            idx = np.array([name_map[n] for n in shared_actions])
            aligned[context][split] = frames[split]["delta"][idx][:, gene_idx][:, selected_local] / scale
    return aligned


def evaluate(aligned):
    names = aligned["names"]
    k_train = aligned["k562"]["train"]
    r_train = aligned["rpe1"]["train"]
    r_val = aligned["rpe1"]["validation"]
    r_test = aligned["rpe1"]["test"]
    replicate = np.diag(cosine_matrix(r_train, r_val))
    keep = replicate >= .30
    tau = float(np.median(replicate[keep]))
    predictions = {
        "Mean": np.repeat(r_train.mean(axis=0, keepdims=True), len(names), axis=0),
        "K562 transfer": k_train,
        "RPE1 in-context": r_train,
    }
    actual = cosine_matrix(r_test, r_test)
    rows, budget_rows = [], []
    for method, pred in predictions.items():
        scores = cosine_matrix(r_test, pred)
        ranks = inverse_ranks(scores, names, names)
        pcc = row_pearson(pred, r_test)
        for i, name in enumerate(names):
            allowed = np.arange(len(names)) != i
            idx = np.flatnonzero(allowed)
            order = idx[np.argsort(-scores[i, idx])]
            chosen = order[0]
            rows.append({
                "method": method, "perturbation": str(name),
                "replicate_consistent": bool(keep[i]), "forward_pearson": float(pcc[i]),
                "closed_rank": int(ranks[i]), "closed_mrr": float(1/ranks[i]),
                "recommended": str(names[chosen]),
                "open_target_cosine": float(actual[i, chosen]),
                "top1_success": int(actual[i, chosen] >= tau),
                "oracle_support": float(actual[i, idx].max()),
                "supported": int(actual[i, idx].max() >= tau),
            })
            for budget in [1,2,3,5,10,20]:
                budget_rows.append({
                    "method": method, "perturbation": str(name), "budget": budget,
                    "success": int(actual[i, order[:budget]].max() >= tau),
                    "best_cosine": float(actual[i, order[:budget]].max()),
                })
    per_query = pd.DataFrame(rows)
    per_query = per_query[per_query.replicate_consistent].reset_index(drop=True)
    budget = pd.DataFrame(budget_rows)
    budget = budget[budget.perturbation.isin(per_query.perturbation.unique())].reset_index(drop=True)
    return per_query, budget, replicate, keep, tau


def summarize(per_query):
    rng = np.random.default_rng(SEED)
    rows = []
    for method in METHODS:
        d = per_query[per_query.method == method]
        for metric in ["forward_pearson", "closed_mrr", "open_target_cosine", "top1_success"]:
            values = d[metric].to_numpy(float)
            boots = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(1500)])
            rows.append({"method": method, "metric": metric, "mean": values.mean(),
                         "ci_low": np.quantile(boots,.025), "ci_high": np.quantile(boots,.975),
                         "n": len(values)})
    return pd.DataFrame(rows)


def summarize_budget(budget):
    rng = np.random.default_rng(SEED + 5)
    rows = []
    for (method, candidate_budget), d in budget.groupby(["method", "budget"]):
        values = d.success.to_numpy(float)
        boots = np.array([
            rng.choice(values, len(values), replace=True).mean() for _ in range(1500)
        ])
        rows.append({
            "method": method, "budget": int(candidate_budget),
            "mean": values.mean(), "ci_low": np.quantile(boots, .025),
            "ci_high": np.quantile(boots, .975), "n": len(values),
        })
    return pd.DataFrame(rows)


def _style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial","Helvetica","DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "axes.linewidth": .65, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def plot_figure(summary, budget_summary, per_query, budget, replicate, keep, tau, counts, outdir):
    _style()
    fig = plt.figure(figsize=(7.15,5.45))
    gs = fig.add_gridspec(2,12,left=.07,right=.985,top=.95,bottom=.10,hspace=.48,wspace=1.0)
    ax_a=fig.add_subplot(gs[0,:3]); ax_b=fig.add_subplot(gs[0,3:8]); ax_c=fig.add_subplot(gs[0,8:])
    ax_d=fig.add_subplot(gs[1,:5]); ax_e=fig.add_subplot(gs[1,5:9]); ax_f=fig.add_subplot(gs[1,9:])
    axes=[ax_a,ax_b,ax_c,ax_d,ax_e,ax_f]
    for ax,label in zip(axes,"abcdef"):
        ax.text(-.10,1.04,label,transform=ax.transAxes,fontsize=8,fontweight="bold")
    ax_a.barh(np.arange(3), [counts[x]/1000 for x in SPLITS], color=["#255B78","#91AFC0","#D48B6A"])
    ax_a.set_yticks(np.arange(3),["Train","Validation","Test"]); ax_a.invert_yaxis()
    ax_a.set(xlabel="Cells (x1,000)",title="RPE1 fixed gem-group split")
    labels={"forward_pearson":"Forward delta Pearson","closed_mrr":"Closed MRR","open_target_cosine":"Open target cosine"}
    x=np.arange(3)
    for method,offset in zip(METHODS,[-.16,0,.16]):
        d=summary.set_index(["method","metric"]).loc[[(method,m) for m in labels]]
        ax_b.errorbar(x+offset,d["mean"],yerr=np.vstack([d["mean"]-d.ci_low,d.ci_high-d["mean"]]),
                      fmt="o",ms=4,capsize=2,color=COLORS[method],label=method)
    ax_b.set_xticks(x,list(labels.values())); ax_b.set_ylim(-.1,1.02)
    ax_b.set(ylabel="Performance",title="Within- versus cross-context retrieval")
    ax_b.legend(fontsize=5.5)
    for method in METHODS:
        vals=np.sort(per_query.query("method==@method").open_target_cosine)
        ax_c.plot(vals,np.arange(1,len(vals)+1)/len(vals),lw=1.3,color=COLORS[method],label=method)
    ax_c.axvline(tau,color=INK,ls="--",lw=.8)
    ax_c.set(xlabel="RPE1 observed target cosine",ylabel="Cumulative fraction",title=f"Independent RPE1 replay (tau={tau:.2f})")
    for method in METHODS:
        d=budget_summary[budget_summary.method==method].sort_values("budget")
        ax_d.plot(d.budget,d["mean"],"o-",ms=3,lw=1.2,color=COLORS[method],label=method)
        ax_d.fill_between(d.budget,d.ci_low,d.ci_high,color=COLORS[method],alpha=.10,linewidth=0)
    ax_d.set_xscale("log"); ax_d.set_xticks([1,2,3,5,10,20],[1,2,3,5,10,20]); ax_d.set_ylim(0,1.02)
    ax_d.set(xlabel="Candidate budget",ylabel="Success fraction",title="Budgeted RPE1 target attainment")
    transfer=per_query.query("method=='K562 transfer'").set_index("perturbation")
    context=per_query.query("method=='RPE1 in-context'").set_index("perturbation")
    pair=transfer[["open_target_cosine"]].join(context[["open_target_cosine"]],lsuffix="_transfer",rsuffix="_context")
    ax_e.scatter(pair.open_target_cosine_context,pair.open_target_cosine_transfer,s=8,alpha=.4,color="#255B78",edgecolor="none")
    ax_e.plot([-1,1],[-1,1],ls="--",color="#AAB1B7",lw=.7)
    ax_e.set(xlim=(-1,1),ylim=(-1,1),xlabel="RPE1 in-context cosine",ylabel="K562-transfer cosine",title="Per-target context shift")
    delta = pair.open_target_cosine_transfer-pair.open_target_cosine_context
    rng = np.random.default_rng(SEED + 11)
    boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
    ax_e.text(
        .04, .95,
        f"mean delta={delta.mean():.3f} [{np.quantile(boot,.025):.3f}, {np.quantile(boot,.975):.3f}]",
        transform=ax_e.transAxes, va="top", fontsize=5.5
    )
    top1=summary.query("metric=='top1_success'").set_index("method").loc[METHODS]
    ax_f.bar(np.arange(3),top1["mean"],color=[COLORS[m] for m in METHODS])
    ax_f.errorbar(
        np.arange(3), top1["mean"],
        yerr=np.vstack([top1["mean"]-top1.ci_low,top1.ci_high-top1["mean"]]),
        fmt="none", ecolor=INK, elinewidth=.7, capsize=2,
    )
    ax_f.set_xticks(np.arange(3),["Mean","K562","RPE1"],rotation=20,ha="right")
    ax_f.set_ylim(0,1.02); ax_f.set(ylabel="Top-1 success fraction",title="One-shot target attainment")
    for i,v in enumerate(top1["mean"]): ax_f.text(i,v+.035,f"{v:.2f}",ha="center",fontsize=6)
    for ax in axes: ax.grid(color=GRID,lw=.45,axis="y" if ax in [ax_a,ax_b,ax_f] else "both")
    outdir.mkdir(parents=True,exist_ok=True)
    stem=outdir/"Figure10_RPE1_context_transfer"
    for ext,kwargs in [("svg",{}),("pdf",{}),("tiff",{"dpi":600}),("png",{"dpi":300})]:
        fig.savefig(stem.with_suffix("."+ext),bbox_inches="tight",**kwargs)
    plt.close(fig)


def run_rpe1_context_transfer(project_root: Path):
    rbase=project_root/"artifacts"/"rpe1_context_transfer"; source=rbase/"source_data"
    source.mkdir(parents=True,exist_ok=True)
    rcent,rmeta,rgenes=aggregate_rpe1(
        project_root/"data/cellnavigator/genetic/replogle_rpe1/rpe1_normalized_singlecell_01.h5ad",
        rbase/"cache"
    )
    kz=np.load(project_root/"artifacts/k562_mvp/cache/k562_centroids_all_genes.npz")
    kmeta=pd.read_csv(project_root/"artifacts/k562_mvp/cache/k562_centroid_metadata.csv")
    kgenes=pd.read_csv(project_root/"artifacts/k562_mvp/cache/k562_genes.csv")
    kframes=all_gene_deltas(kz["centroids"],kmeta)
    rframes=all_gene_deltas(rcent,rmeta)
    aligned=align_contexts(kframes,kgenes,rframes,rgenes)
    per_query,budget,replicate,keep,tau=evaluate(aligned)
    summary=summarize(per_query)
    budget_summary=summarize_budget(budget)
    counts=rmeta.groupby("split").n_cells.sum().reindex(SPLITS).to_dict()
    per_query.to_csv(source/"per_query_results.csv",index=False)
    budget.to_csv(source/"budget_results.csv",index=False)
    budget_summary.to_csv(source/"budget_summary.csv",index=False)
    summary.to_csv(source/"summary_metrics.csv",index=False)
    pd.DataFrame({"perturbation":aligned["names"],"rpe1_train_validation_cosine":replicate,
                  "query_whitelist":keep}).to_csv(source/"query_whitelist.csv",index=False)
    pd.DataFrame({"gene_id":aligned["genes"]}).to_csv(source/"aligned_features.csv",index=False)
    protocol={"rpe1_split":{"train":[1,36],"validation":[37,46],"test":[47,56]},
              "minimum_cells_per_condition_per_split":10,"shared_actions":len(aligned["names"]),
              "aligned_features":len(aligned["genes"]),"query_count":int(keep.sum()),
              "query_rule":"RPE1 train-validation self cosine >= 0.30",
              "support_threshold":tau,"threshold_rule":"median whitelisted RPE1 train-validation self cosine"}
    (rbase/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    plot_figure(summary,budget_summary,per_query,budget,replicate,keep,tau,counts,rbase/"figures")
    table=summary.pivot(index="method",columns="metric",values="mean")
    transfer=per_query.query("method=='K562 transfer'").set_index("perturbation")
    context=per_query.query("method=='RPE1 in-context'").set_index("perturbation")
    paired=transfer[["open_target_cosine"]].join(
        context[["open_target_cosine"]],lsuffix="_transfer",rsuffix="_context"
    )
    delta=paired.open_target_cosine_transfer-paired.open_target_cosine_context
    rng=np.random.default_rng(SEED+12)
    boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(5000)])
    statistic,pvalue=wilcoxon(
        paired.open_target_cosine_transfer,
        paired.open_target_cosine_context,
        alternative="two-sided",
    )
    comparison=pd.DataFrame([{
        "comparison":"K562 transfer minus RPE1 in-context",
        "n":len(delta),"mean_difference":delta.mean(),
        "ci_low":np.quantile(boot,.025),"ci_high":np.quantile(boot,.975),
        "fraction_transfer_lower":float((delta<0).mean()),
        "wilcoxon_statistic":float(statistic),"wilcoxon_pvalue":float(pvalue),
    }])
    comparison.to_csv(source/"paired_context_comparison.csv",index=False)
    report=(
        "# Replogle RPE1 cross-context validation\n\n"
        f"- Shared K562/RPE1 actions: {len(aligned['names']):,}.\n"
        f"- Aligned expression features: {len(aligned['genes']):,}.\n"
        f"- Replicate-consistent RPE1 queries: {int(keep.sum()):,}.\n"
        f"- RPE1 in-context open cosine: {table.loc['RPE1 in-context','open_target_cosine']:.3f}.\n"
        f"- K562-to-RPE1 transfer open cosine: {table.loc['K562 transfer','open_target_cosine']:.3f}.\n"
        f"- Paired transfer difference: {delta.mean():.3f} (bootstrap 95% CI {np.quantile(boot,.025):.3f} to {np.quantile(boot,.975):.3f}; two-sided Wilcoxon P={pvalue:.3g}).\n"
        f"- Fraction of targets with lower transfer attainment: {(delta<0).mean():.3f}.\n"
        "- Candidate rankings use training partitions only; all reported target attainment uses RPE1 test gem groups.\n"
    )
    (rbase/"RPE1_CONTEXT_TRANSFER_REPORT.md").write_text(report,encoding="utf-8")
    print(report)
