import numpy as np
import dimod
from dimod import SimulatedAnnealingSampler
from dwave.samplers import PathIntegralAnnealingSampler
from parameter_optimization import create_qubo
from typing import List, Dict, Tuple, Set, FrozenSet, Optional, Any
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import matplotlib.ticker as ticker
import random as rng_mod



def run_sa(
    qubo: Dict[Tuple[int, int], float],
    num_reads: int,
    num_sweeps: int,
    beta_range: Tuple[float, float] = (0.1, 4.0),
    seed: Optional[int] = None,
) -> dimod.SampleSet:
    """Run classical Simulated Annealing via dimod."""
    sampler = SimulatedAnnealingSampler()
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
    return sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_range=beta_range,
        seed=seed,
    )


def run_sqa(
    qubo: Dict[Tuple[int, int], float],
    num_reads: int,
    num_sweeps: int,
    beta_range: Tuple[float, float] = (0.1, 4.0),
    num_sweeps_per_beta: int = 2,
    seed: Optional[int] = None,
) -> dimod.SampleSet:
    """Run Simulated Quantum Annealing via D-Wave's PathIntegralAnnealingSampler."""
    sampler = PathIntegralAnnealingSampler()
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

    effective_sweeps = max(
        num_sweeps_per_beta,
        (num_sweeps // num_sweeps_per_beta) * num_sweeps_per_beta,
    )

    return sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=effective_sweeps,
        num_sweeps_per_beta=num_sweeps_per_beta,
        beta_range=beta_range,
        seed=seed,
    )


def compute_sampler_metrics(
    sampleset: dimod.SampleSet,
    drug_ids: List[str],
    gt_combinations: Set[FrozenSet],
    ground_energy: Optional[float] = None,
) -> Dict:
    """
    Compute comparison metrics from a sample set.
    """
    drug_set = set(drug_ids)
    valid_gt = {c for c in gt_combinations if c.issubset(drug_set)}

    energies = []
    matches = []
    found_valid = set()

    for sample, energy in zip(sampleset.samples(), sampleset.record.energy):
        selected = frozenset(drug_ids[i] for i, v in sample.items() if v == 1)
        energies.append(float(energy))
        is_match = selected in valid_gt
        matches.append(is_match)
        if is_match:
            found_valid.add(selected)

    energies = np.array(energies)
    matches = np.array(matches)
    n_reads = len(energies)
    best_e = float(np.min(energies))

    result = {
        "match_rate": float(np.mean(matches)),
        "n_unique_valid": len(found_valid),
        "unique_valid_set": found_valid,
        "best_energy": best_e,
        "mean_energy": float(np.mean(energies)),
        "ground_prob": float(np.sum(np.abs(energies - best_e) < 1e-6) / n_reads),
    }

    if ground_energy is not None:
        result["approx_ratio"] = float(np.mean(energies / ground_energy))

    return result


def compare_sa_sqa_vs_sweeps(
    qubo: Dict[Tuple[int, int], float],
    drug_ids: List[str],
    gt_combinations: Set[FrozenSet],
    sweep_values: List[int],
    num_reads: int = 1024,
    beta_range: Tuple[float, float] = (0.1, 4.0),
    num_repeats: int = 5,
    sqa_sweeps_per_beta: int = 2,
    seed: int = 42,
    ground_energy: Optional[float] = None,
) -> Dict:
    """
    Compare SA and SQA at varying numbers of sweeps.

    For each sweep count, both samplers are run num_repeats times.
    All metrics are averaged over repeats.
    """
    results = {"sa": {}, "sqa": {}}

    for ns in sweep_values:
        print(f"\n  num_sweeps = {ns}")

        for method in ["sa", "sqa"]:
            metrics_list = []
            times = []

            for r in range(num_repeats):
                rseed = seed + r * 1000 + ns

                t0 = time.time()
                if method == "sa":
                    ss = run_sa(qubo, num_reads, ns, beta_range, seed=rseed)
                else:
                    ss = run_sqa(qubo, num_reads, ns, beta_range,
                                 sqa_sweeps_per_beta, seed=rseed)
                elapsed = time.time() - t0
                times.append(elapsed)

                m = compute_sampler_metrics(ss, drug_ids, gt_combinations,
                                           ground_energy)
                metrics_list.append(m)

            agg = {}
            for key in ["match_rate", "n_unique_valid", "best_energy",
                        "mean_energy", "ground_prob"]:
                vals = [m[key] for m in metrics_list]
                agg[key] = (float(np.mean(vals)), float(np.std(vals)))
            agg["wall_time"] = (float(np.mean(times)), float(np.std(times)))

            results[method][ns] = agg
            mr = agg["match_rate"]
            nuv = agg["n_unique_valid"]
            print(f"    {method.upper():3s}: match_rate={mr[0]:.3f}±{mr[1]:.3f}, "
                  f"unique_valid={nuv[0]:.1f}±{nuv[1]:.1f}, "
                  f"time={agg['wall_time'][0]:.2f}s")

    return results


def compare_sa_sqa_scaling(
    z_array: np.ndarray,
    s_matrix: np.ndarray,
    gamma: float,
    beta: float,
    core_indices: List[int],
    padding_indices: List[int],
    drug_ids: List[str],
    gt_combinations: Set[FrozenSet],
    sizes: List[int],
    num_sweeps: int = 1000,
    num_reads: int = 1024,
    num_repeats: int = 5,
    num_subproblems: int = 5,
    allowed_sizes: List[int] = [2, 3],
    beta_range: Tuple[float, float] = (0.1, 4.0),
    sqa_sweeps_per_beta: int = 2,
    seed: int = 42,
) -> Dict:
    """
    Compare SA and SQA across problem sizes at a fixed sweep budget.

    For each size n, random subproblems are drawn and both samplers are run. Metrics are averaged over subproblems and repeats.
    """
    import random as rng_mod
    rng = rng_mod.Random(seed)
    n_core = len(core_indices)
    results = {}

    for n in sizes:
        print(f"\n  n = {n}")
        sa_metrics_all = []
        sqa_metrics_all = []

        for sp in range(num_subproblems):
            if n <= n_core:
                subset = sorted(rng.sample(core_indices, n))
            else:
                n_pad = n - n_core
                if n_pad > len(padding_indices):
                    print(f"    [Warn] n={n} too large, skipping")
                    break
                pad_sample = rng.sample(padding_indices, n_pad)
                subset = sorted(core_indices + pad_sample)

            z_sub = z_array[np.array(subset)]
            s_sub = s_matrix[np.ix_(subset, subset)]
            sub_qubo = create_qubo(z_sub, s_sub, gamma, beta, allowed_sizes)
            sub_drug_ids = [drug_ids[i] for i in subset]

            sub_drug_set = set(sub_drug_ids)
            sub_gt = {c for c in gt_combinations if c.issubset(sub_drug_set)}
            if not sub_gt:
                continue

            for r in range(num_repeats):
                rseed = seed + sp * 10000 + r * 100 + n

                ss_sa = run_sa(sub_qubo, num_reads, num_sweeps, beta_range,
                               seed=rseed)
                m_sa = compute_sampler_metrics(ss_sa, sub_drug_ids, sub_gt)
                sa_metrics_all.append(m_sa)

                ss_sqa = run_sqa(sub_qubo, num_reads, num_sweeps, beta_range,
                                 sqa_sweeps_per_beta, seed=rseed + 1)
                m_sqa = compute_sampler_metrics(ss_sqa, sub_drug_ids, sub_gt)
                sqa_metrics_all.append(m_sqa)

        size_result = {}
        for label, mlist in [("sa", sa_metrics_all), ("sqa", sqa_metrics_all)]:
            if not mlist:
                continue
            agg = {}
            for key in ["match_rate", "n_unique_valid", "best_energy",
                        "mean_energy"]:
                vals = [m[key] for m in mlist]
                agg[key] = (float(np.mean(vals)), float(np.std(vals)))
            size_result[label] = agg

        if size_result:
            results[n] = size_result
            for label in ["sa", "sqa"]:
                if label in size_result:
                    mr = size_result[label]["match_rate"]
                    print(f"    {label.upper():3s}: match_rate="
                          f"{mr[0]:.3f}±{mr[1]:.3f}")

    return results


RCPARAMS = {
    "font.family": "sans-serif",
    "font.size": 12,           
    "axes.titlesize": 14,      
    "axes.labelsize": 14,      
    "legend.fontsize": 11,     
    "xtick.labelsize": 12,     
    "ytick.labelsize": 12,     
    "figure.dpi": 300,
}

def plot_sweep_comparison(
    results: Dict,
    metric: str = "match_rate",
    ylabel: Optional[str] = None,
    title: str = "",
    figsize: Tuple = (5.5, 4.0),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a metric as a function of num_sweeps for SA vs SQA.
    """
    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=figsize)

    for method, color, marker in [("sa", "#d62728", "s"), ("sqa", "#1f77b4", "o")]:
        sweeps = sorted(results[method].keys())
        means = [results[method][ns][metric][0] for ns in sweeps]
        stds = [results[method][ns][metric][1] for ns in sweeps]

        label = "SA" if method == "sa" else "SQA"
        ax.errorbar(
            sweeps, means, yerr=stds,
            fmt=f"{marker}-", color=color, label=label,
            markersize=5, capsize=3, linewidth=1.5,
            markeredgecolor="black", markeredgewidth=0.4,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of sweeps")
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
    return fig


def plot_scaling_comparison(
    results: Dict,
    metric: str = "match_rate",
    ylabel: Optional[str] = None,
    title: str = "",
    figsize: Tuple = (5.5, 4.0),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a metric as a function of problem size n for SA vs SQA
    at fixed sweep budget.
    """
    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=figsize)

    for method, color, marker in [("sa", "#d62728", "s"), ("sqa", "#1f77b4", "o")]:
        sizes = sorted(n for n in results if method in results[n])
        means = [results[n][method][metric][0] for n in sizes]
        stds = [results[n][method][metric][1] for n in sizes]

        label = "SA" if method == "sa" else "SQA"
        ax.errorbar(
            sizes, means, yerr=stds,
            fmt=f"{marker}-", color=color, label=label,
            markersize=5, capsize=3, linewidth=1.5,
            markeredgecolor="black", markeredgewidth=0.4,
        )

    ax.set_xlabel("Number of drugs $n$")
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
    return fig


def plot_multi_metric_sweep_comparison(
    results: Dict,
    title: str = "",
    figsize: Tuple = (12, 4.0),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Three-panel figure: match_rate, n_unique_valid, and best_energy
    vs num_sweeps for SA and SQA.
    """
    plt.rcParams.update(RCPARAMS)

    metrics = [
        ("match_rate", "Validated match rate"),
        ("n_unique_valid", "Unique validated found"),
        ("best_energy", "Best energy"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (metric, ylabel) in zip(axes, metrics):
        for method, color, marker in [("sa", "#d62728", "s"),
                                       ("sqa", "#1f77b4", "o")]:
            sweeps = sorted(results[method].keys())
            means = [results[method][ns][metric][0] for ns in sweeps]
            stds = [results[method][ns][metric][1] for ns in sweeps]

            label = "SA" if method == "sa" else "SQA"
            ax.errorbar(
                sweeps, means, yerr=stds,
                fmt=f"{marker}-", color=color, label=label,
                markersize=4, capsize=2, linewidth=1.3,
                markeredgecolor="black", markeredgewidth=0.3,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Number of sweeps")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=True, framealpha=0.9, edgecolor="black", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.15, linestyle="--", linewidth=0.5)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
    return fig



def compute_metrics(
    sampleset: dimod.SampleSet,
    drug_ids: List[str],
    gt_combinations: Set[FrozenSet],
) -> Dict:
    """
    Compute comparison metrics from a sample set.
    """
    drug_set = set(drug_ids)
    valid_gt = {c for c in gt_combinations if c.issubset(drug_set)}

    matches = 0
    found_valid = set()
    energies = []

    for sample, energy in zip(sampleset.samples(), sampleset.record.energy):
        selected = frozenset(drug_ids[i] for i, v in sample.items() if v == 1)
        e = float(energy)
        energies.append(e)
        if selected in valid_gt:
            matches += 1
            found_valid.add(selected)

    n_reads = len(energies)
    p_s = matches / n_reads if n_reads > 0 else 0.0

    return {
        "p_s": p_s,
        "match_rate": p_s,
        "n_unique_valid": len(found_valid),
        "unique_valid_set": found_valid,
        "best_energy": float(np.min(energies)) if energies else np.inf,
        "mean_energy": float(np.mean(energies)) if energies else np.inf,
    }


def compute_tts(p_s: float, num_sweeps: int, target_prob: float = 0.99) -> float:
    """
    Time-to-solution in units of sweeps.
    TTS_p = num_sweeps * ceil( log(1-p) / log(1-p_s) )

    Returns np.inf if p_s == 0.
    """
    if p_s <= 0.0:
        return np.inf
    if p_s >= 1.0:
        return float(num_sweeps)
    return num_sweeps * np.ceil(np.log(1.0 - target_prob) / np.log(1.0 - p_s))

def compare_vs_sweeps(
    qubo: Dict[Tuple[int, int], float],
    drug_ids: List[str],
    gt_combinations: Set[FrozenSet],
    sweep_values: List[int],
    num_reads: int = 1024,
    num_repeats: int = 5,
    beta_range: Tuple[float, float] = (0.1, 4.0),
    seed: int = 42,
) -> Dict:
    """
    Compare SA and SQA at varying sweep budgets on a fixed QUBO.
    Returns dict with 'sa' and 'sqa' keys, each mapping
    sweep_count -> {metric: (mean, std)}.
    """
    results = {"sa": {}, "sqa": {}}

    for ns in sweep_values:
        print(f"\n  sweeps = {ns}")

        for method in ["sa", "sqa"]:
            metrics_list = []
            wall_times = []

            for r in range(num_repeats):
                rseed = seed + r * 1000 + ns
                
                t0 = time.time()
                if method == "sa":
                    ss = run_sa(qubo, num_reads, ns, beta_range, seed=rseed)
                else:
                    ss = run_sqa(qubo, num_reads, ns, beta_range, seed=rseed)
                wt = time.time() - t0
                wall_times.append(wt)

                m = compute_metrics(ss, drug_ids, gt_combinations)
                m["tts_sweeps"] = compute_tts(m["p_s"], ns)
                m["tts_wall"] = compute_tts(m["p_s"], 1) * wt
                metrics_list.append(m)

            agg = {}
            for key in ["match_rate", "p_s", "n_unique_valid",
                        "best_energy", "mean_energy",
                        "tts_sweeps", "tts_wall"]:
                vals = [m[key] for m in metrics_list]
                if "tts" in key:
                    finite = [v for v in vals if np.isfinite(v)]
                    if finite:
                        agg[key] = (float(np.median(finite)),
                                    float(np.std(finite)))
                    else:
                        agg[key] = (np.inf, 0.0)
                else:
                    agg[key] = (float(np.mean(vals)), float(np.std(vals)))
            agg["wall_time"] = (float(np.mean(wall_times)),
                                float(np.std(wall_times)))

            results[method][ns] = agg
            mr = agg["match_rate"]
            tts = agg["tts_sweeps"]
            nuv = agg["n_unique_valid"]
            print(f"    {method.upper():3s}: p_s={mr[0]:.3f}±{mr[1]:.3f}, "
                  f"TTS={tts[0]:.0f}±{tts[1]:.0f} sweeps, "
                  f"unique={nuv[0]:.1f}±{nuv[1]:.1f}")

    return results

def compare_scaling(
    z_array: np.ndarray,
    s_matrix: np.ndarray,
    gamma: float,
    beta: float,
    core_indices: List[int],
    padding_indices: List[int],
    drug_ids: List[str],
    gt_combinations: Set[FrozenSet],
    sizes: List[int],
    num_sweeps: int = 1000,
    num_reads: int = 1024,
    num_repeats: int = 3,
    num_subproblems: int = 5,
    allowed_sizes: List[int] = [2, 3],
    beta_range: Tuple[float, float] = (0.1, 4.0),
    seed: int = 42,
) -> Dict:
    """
    Compare SA and SQA across problem sizes at a fixed sweep budget.
    Returns dict: n -> {'sa': {...}, 'sqa': {...}}.
    """
    rng = rng_mod.Random(seed)
    n_core = len(core_indices)
    results = {}

    for n in sizes:
        print(f"\n  n = {n}")
        method_metrics = {"sa": [], "sqa": []}

        for sp in range(num_subproblems):
            if n <= n_core:
                subset = sorted(rng.sample(core_indices, n))
            else:
                n_pad = n - n_core
                if n_pad > len(padding_indices):
                    print(f"    [Warn] n={n} too large, skipping")
                    break
                pad_sample = rng.sample(padding_indices, n_pad)
                subset = sorted(core_indices + pad_sample)

            z_sub = z_array[np.array(subset)]
            s_sub = s_matrix[np.ix_(subset, subset)]
            sub_qubo = create_qubo(z_sub, s_sub, gamma, beta, allowed_sizes)
            sub_drug_ids = [drug_ids[i] for i in subset]

            sub_drug_set = set(sub_drug_ids)
            sub_gt = {c for c in gt_combinations if c.issubset(sub_drug_set)}
            if not sub_gt:
                continue

            for r in range(num_repeats):
                rseed = seed + sp * 10000 + r * 100 + n

                ss_sa = run_sa(sub_qubo, num_reads, num_sweeps,
                               beta_range, seed=rseed)
                m_sa = compute_metrics(ss_sa, sub_drug_ids, sub_gt)
                m_sa["tts_sweeps"] = compute_tts(m_sa["p_s"], num_sweeps)
                method_metrics["sa"].append(m_sa)

                ss_sqa = run_sqa(sub_qubo, num_reads, num_sweeps,
                                 beta_range, seed=rseed + 1)
                m_sqa = compute_metrics(ss_sqa, sub_drug_ids, sub_gt)
                m_sqa["tts_sweeps"] = compute_tts(m_sqa["p_s"], num_sweeps)
                method_metrics["sqa"].append(m_sqa)

        size_result = {}
        for label in ["sa", "sqa"]:
            mlist = method_metrics[label]
            if not mlist:
                continue
            agg = {}
            for key in ["match_rate", "n_unique_valid", "best_energy",
                        "mean_energy", "tts_sweeps"]:
                vals = [m[key] for m in mlist]
                if "tts" in key:
                    finite = [v for v in vals if np.isfinite(v)]
                    if finite:
                        agg[key] = (float(np.median(finite)),
                                    float(np.std(finite)))
                    else:
                        agg[key] = (np.inf, 0.0)
                else:
                    agg[key] = (float(np.mean(vals)), float(np.std(vals)))
            size_result[label] = agg

        if size_result:
            results[n] = size_result
            for label in ["sa", "sqa"]:
                if label in size_result:
                    mr = size_result[label]["match_rate"]
                    tts = size_result[label]["tts_sweeps"]
                    print(f"    {label.upper():3s}: p_s={mr[0]:.3f}±{mr[1]:.3f}, "
                          f"TTS={tts[0]:.0f} sweeps")

    return results

def plot_tts_vs_sweeps(
    results: Dict,
    title: str = "",
    figsize: Tuple = (5.5, 4.0),
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    for method, color, marker, label in [
        ("sa",  "#d62728", "s", "SA"),
        ("sqa", "#1f77b4", "o", "SQA"),
    ]:
        sweeps = sorted(results[method].keys())
        tts_means, tts_stds, valid_sweeps = [], [], []

        for ns in sweeps:
            m, s = results[method][ns]["tts_sweeps"]
            if np.isfinite(m):
                valid_sweeps.append(ns)
                tts_means.append(m)
                tts_stds.append(s)

        if valid_sweeps:
            ax.errorbar(
                valid_sweeps, tts_means, yerr=tts_stds,
                fmt=f"{marker}-", color=color, label=label,
                markersize=5, capsize=3, linewidth=1.5,
                markeredgecolor="black", markeredgewidth=0.4,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sweep budget per read")
    ax.set_ylabel(r"$\mathrm{TTS}_{99}$")
    if title:
        ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
    return fig

def plot_tts_scaling(
    results: Dict,
    title: str = "",
    figsize: Tuple = (5.5, 4.0),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot TTS vs problem size n at fixed sweep budget."""
    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=figsize)

    for method, color, marker, label in [
        ("sa", "#d62728", "s", "SA"),
        ("sqa", "#1f77b4", "o", "SQA"),
    ]:
        sizes = sorted(n for n in results if method in results[n])
        means, stds, valid_n = [], [], []
        for n in sizes:
            m, s = results[n][method]["tts_sweeps"]
            if np.isfinite(m):
                valid_n.append(n)
                means.append(m)
                stds.append(s)

        if valid_n:
            ax.errorbar(
                valid_n, means, yerr=stds,
                fmt=f"{marker}-", color=color, label=label,
                markersize=5, capsize=3, linewidth=1.5,
                markeredgecolor="black", markeredgewidth=0.4,
            )

    ax.set_yscale("log")
    ax.set_xlabel("Number of drugs $n$")
    ax.set_ylabel(r"$\mathrm{TTS}_{99}$")
    if title:
        ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
    return fig


def plot_match_rate_scaling(
    results: Dict,
    title: str = "",
    figsize: Tuple = (5.5, 4.0),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot match rate vs problem size n at fixed sweep budget."""
    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=figsize)

    for method, color, marker, label in [
        ("sa", "#d62728", "s", "SA"),
        ("sqa", "#1f77b4", "o", "SQA"),
    ]:
        sizes = sorted(n for n in results if method in results[n])
        means = [results[n][method]["match_rate"][0] for n in sizes]
        stds = [results[n][method]["match_rate"][1] for n in sizes]

        ax.errorbar(
            sizes, means, yerr=stds,
            fmt=f"{marker}-", color=color, label=label,
            markersize=5, capsize=3, linewidth=1.5,
            markeredgecolor="black", markeredgewidth=0.4,
        )

    ax.set_xlabel("Number of drugs $n$")
    ax.set_ylabel("Validated match rate $p_s$")
    if title:
        ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
    return fig


def poly_model(n, a, b):
    """log(TTS) = a * log(n) + b  →  TTS ~ n^a"""
    return a * np.log(n) + b

def exp_model(n, a, b):
    """log(TTS) = a * n + b  →  TTS ~ exp(a*n)"""
    return a * n + b

def fit_scaling(sizes, tts_values, tts_stds=None):
    """
    Fit both polynomial and exponential models to TTS(n).
    Returns dict with fit parameters, R^2, and labels.
    """
    n = np.array(sizes, dtype=float)
    log_tts = np.log(tts_values)

    if tts_stds is not None:
        weights = tts_values / np.maximum(tts_stds, 1e-10)
    else:
        weights = None

    results = {}

    try:
        popt_p, pcov_p = curve_fit(poly_model, n, log_tts, sigma=1.0/weights if weights is not None else None)
        pred_p = poly_model(n, *popt_p)
        ss_res_p = np.sum((log_tts - pred_p)**2)
        ss_tot = np.sum((log_tts - np.mean(log_tts))**2)
        r2_p = 1 - ss_res_p / ss_tot if ss_tot > 0 else 0
        results['poly'] = {
            'params': popt_p,
            'cov': pcov_p,
            'r2': r2_p,
            'label': f'$n^{{{popt_p[0]:.2f}}}$ ($R^2={r2_p:.3f}$)',
            'exponent': popt_p[0],
        }
    except Exception as e:
        print(f"  Polynomial fit failed: {e}")

    try:
        popt_e, pcov_e = curve_fit(exp_model, n, log_tts, sigma=1.0/weights if weights is not None else None)
        pred_e = exp_model(n, *popt_e)
        ss_res_e = np.sum((log_tts - pred_e)**2)
        r2_e = 1 - ss_res_e / ss_tot if ss_tot > 0 else 0
        results['exp'] = {
            'params': popt_e,
            'cov': pcov_e,
            'r2': r2_e,
            'label': f'$e^{{{popt_e[0]:.3f} n}}$ ($R^2={r2_e:.3f}$)',
            'rate': popt_e[0],
        }
    except Exception as e:
        print(f"  Exponential fit failed: {e}")

    return results


def analyze_scaling(scaling_results):
    """
    Extract TTS data from scaling_results and fit both models for SA and SQA. 
    Returns dict with sizes, TTS, stds, and fit results for each method.
    """
    analysis = {}

    for method in ['sa', 'sqa']:
        sizes, tts, tts_std = [], [], []
        for n in sorted(scaling_results.keys()):
            if method not in scaling_results[n]:
                continue
            m, s = scaling_results[n][method]['tts_sweeps']
            if np.isfinite(m) and m > 0:
                sizes.append(n)
                tts.append(m)
                tts_std.append(s)

        if len(sizes) < 3:
            print(f"  {method.upper()}: insufficient finite TTS points ({len(sizes)}), skipping fits")
            analysis[method] = {'sizes': sizes, 'tts': tts, 'tts_std': tts_std, 'fits': {}}
            continue

        sizes_arr = np.array(sizes)
        tts_arr = np.array(tts)
        std_arr = np.array(tts_std)

        print(f"\n  {method.upper()} scaling fits:")
        fits = fit_scaling(sizes_arr, tts_arr, std_arr)
        for name, f in fits.items():
            print(f"    {name}: {f['label']}")

        analysis[method] = {
            'sizes': sizes_arr,
            'tts': tts_arr,
            'tts_std': std_arr,
            'fits': fits,
        }

    return analysis

def plot_tts_scaling_with_fits(
    analysis,
    title="",
    figsize=(6, 4.5),
    save_path=None,
):
    """
    Plot TTS vs n with fitted polynomial scaling curves for SA and SQA.
    """
    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=figsize)

    method_style = {
        'sa':  {'color': '#d62728', 'marker': 's', 'label': 'SA'},
        'sqa': {'color': '#1f77b4', 'marker': 'o', 'label': 'SQA'},
    }

    n_dense = np.linspace(3, 55, 200)

    for method in ['sa', 'sqa']:
        if method not in analysis:
            continue
        d = analysis[method]
        if len(d['sizes']) == 0:
            continue

        sty = method_style[method]

        ax.errorbar(
            d['sizes'], d['tts'], yerr=d['tts_std'],
            fmt=f"{sty['marker']}", color=sty['color'], label=sty['label'],
            markersize=6, capsize=3,
            markeredgecolor='black', markeredgewidth=0.4, zorder=5,
        )

        fits = d['fits']
        
        if not fits or 'poly' not in fits:
            continue

        fdata = fits['poly']
        y_fit = np.exp(poly_model(n_dense, *fdata['params']))

        fit_label = f"{sty['label']} fit: {fdata['label']}"
        ax.plot(
            n_dense, y_fit,
            linestyle='--', color=sty['color'],
            linewidth=1.5, alpha=0.7, label=fit_label, zorder=3,
        )

    ax.set_yscale('log')
    ax.set_xlabel('Number of drugs $n$')
    ax.set_ylabel(r'$\mathrm{TTS}_{99}$')
    if title:
        ax.set_title(title)

    ax.legend(frameon=True, framealpha=0.9, edgecolor='black', loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.15, linestyle='--', linewidth=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
    return fig


def plot_match_rate_scaling_with_fits(
    scaling_results,
    title="",
    figsize=(6, 4.5),
    save_path=None,
):
    """
    Plot match rate p_s vs n for SA and SQA).
    Complementary to the TTS plot.
    """
    plt.rcParams.update(RCPARAMS)
    fig, ax = plt.subplots(figsize=figsize)

    for method, color, marker, label in [
        ('sa', '#d62728', 's', 'SA'),
        ('sqa', '#1f77b4', 'o', 'SQA'),
    ]:
        sizes, means, stds = [], [], []
        for n in sorted(scaling_results.keys()):
            if method in scaling_results[n]:
                sizes.append(n)
                m, s = scaling_results[n][method]['match_rate']
                means.append(m)
                stds.append(s)

        ax.errorbar(
            sizes, means, yerr=stds,
            fmt=f'{marker}-', color=color, label=label,
            markersize=5, capsize=3, linewidth=1.5,
            markeredgecolor='black', markeredgewidth=0.4,
        )

    ax.set_xlabel('Number of drugs $n$')
    ax.set_ylabel('Validated match rate $p_s$')
    if title:
        ax.set_title(title)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.15, linestyle='--', linewidth=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
    return fig


def print_scaling_summary(analysis):
    """Print a formatted summary table of the scaling analysis."""
    print(f"\n{'='*70}")
    print(f"  SCALING ANALYSIS SUMMARY")
    print(f"{'='*70}")

    for method in ['sa', 'sqa']:
        if method not in analysis:
            continue
        d = analysis[method]
        print(f"\n  {method.upper()}")
        print(f"  {'n':>4s}  {'TTS (sweeps)':>14s}  {'± std':>10s}")
        print(f"  {'─'*32}")
        for n, t, s in zip(d['sizes'], d['tts'], d['tts_std']):
            print(f"  {int(n):4d}  {t:14.1f}  {s:10.1f}")

        fits = d['fits']
        if fits:
            print(f"\n  Fits:")
            for fname, fdata in fits.items():
                tag = "Polynomial" if fname == "poly" else "Exponential"
                print(f"    {tag:12s}: {fdata['label']}")

            best = max(fits.keys(), key=lambda k: fits[k]['r2'])
            print(f"    Best fit: {best} (R² = {fits[best]['r2']:.4f})")


def full_scaling_report(all_scaling, verbose=True):
    """
    From the all_scaling dict, compute polynomial and exponential fits
    with 95% confidence intervals for all diseases and both methods.
    """
    report = {}

    for dname, sr in all_scaling.items():
        if verbose:
            print(f"\n{'=' * 70}\n  {dname}\n{'=' * 70}")
        report[dname] = {}

        for method in ["sa", "sqa"]:
            sizes, tts, stds = [], [], []
            for n in sorted(sr.keys()):
                if method in sr[n]:
                    m, s = sr[n][method]["tts_sweeps"]
                    if np.isfinite(m) and m > 0:
                        sizes.append(n)
                        tts.append(m)
                        stds.append(s)

            sizes = np.array(sizes, dtype=float)
            tts   = np.array(tts,   dtype=float)
            stds  = np.array(stds,  dtype=float)
            log_tts = np.log(tts)
            ss_tot = np.sum((log_tts - np.mean(log_tts)) ** 2)

            popt_p, pcov_p = curve_fit(poly_model, sizes, log_tts)
            alpha     = popt_p[0]
            alpha_se  = np.sqrt(pcov_p[0, 0])
            alpha_ci  = alpha_se * 1.96
            r2_p      = 1 - np.sum((log_tts - poly_model(sizes, *popt_p)) ** 2) / ss_tot

            popt_e, pcov_e = curve_fit(exp_model, sizes, log_tts)
            c_exp, c_se = popt_e[0], np.sqrt(pcov_e[0, 0])
            c_ci  = c_se * 1.96
            r2_e  = 1 - np.sum((log_tts - exp_model(sizes, *popt_e)) ** 2) / ss_tot

            report[dname][method] = {
                "sizes":   sizes,
                "tts":     tts,
                "tts_std": stds,
                "poly": {"alpha": alpha, "alpha_ci": alpha_ci, "alpha_se": alpha_se,
                         "r2": r2_p, "params": popt_p, "cov": pcov_p},
                "exp":  {"c": c_exp, "c_ci": c_ci, "c_se": c_se,
                         "r2": r2_e, "params": popt_e, "cov": pcov_e},
            }

            if verbose:
                print(f"\n  {method.upper()}:")
                print(f"    Poly:  alpha = {alpha:.2f} ± {alpha_ci:.2f}  (R² = {r2_p:.3f})")
                print(f"    Exp:   c     = {c_exp:.4f} ± {c_ci:.4f}  (R² = {r2_e:.3f})")

        if verbose:
            sa_alpha, sa_ci   = report[dname]["sa"]["poly"]["alpha"],  report[dname]["sa"]["poly"]["alpha_ci"]
            sqa_alpha, sqa_ci = report[dname]["sqa"]["poly"]["alpha"], report[dname]["sqa"]["poly"]["alpha_ci"]
            sa_lo,  sa_hi  = sa_alpha - sa_ci,   sa_alpha + sa_ci
            sqa_lo, sqa_hi = sqa_alpha - sqa_ci, sqa_alpha + sqa_ci
            overlap = (sa_lo <= sqa_hi) and (sqa_lo <= sa_hi)
            print(f"\n  CI overlap check:\n    SA:  [{sa_lo:.2f}, {sa_hi:.2f}]"
                  f"\n    SQA: [{sqa_lo:.2f}, {sqa_hi:.2f}]\n    Overlap: {overlap}")
            if overlap:
                print("    -> Cannot claim scaling advantage for either method")
            else:
                better = "SA" if sa_alpha < sqa_alpha else "SQA"
                print(f"    -> Consistent with scaling advantage for {better}")

    return report


def _exact_sci_notation(x, pos):
    if x <= 0:
        return ""
    exponent = int(np.floor(np.log10(x)))
    coeff = round(x / (10 ** exponent), 1)
    if coeff == 1.0:
        return r"$10^{{{}}}$".format(exponent)
    return r"${:g} \times 10^{{{}}}$".format(coeff, exponent)


def plot_tts_scaling_final(report_disease, figsize=(5.5, 4.0), save_path=None):
    """Plot a single disease's TTS scaling with polynomial fit and CI band."""
    fig, ax = plt.subplots(figsize=figsize)
    n_dense = np.linspace(8, 55, 200)

    method_style = {
        "sa":  {"color": "#d62728", "marker": "s", "label": "SA"},
        "sqa": {"color": "#1f77b4", "marker": "o", "label": "SQA"},
    }

    for method in ["sa", "sqa"]:
        d = report_disease[method]
        sty = method_style[method]

        ax.errorbar(
            d["sizes"], d["tts"], yerr=d.get("tts_std", None),
            fmt=sty["marker"], color=sty["color"],
            markersize=5, capsize=3,
            markeredgecolor="black", markeredgewidth=0.4, zorder=5,
            label=sty["label"],
        )

        p = d["poly"]
        alpha, b = p["params"]
        alpha_se = p["alpha_se"]
        y_fit   = np.exp(poly_model(n_dense, alpha,            b))
        y_upper = np.exp(poly_model(n_dense, alpha + alpha_se, b))
        y_lower = np.exp(poly_model(n_dense, alpha - alpha_se, b))

        ci_str = f"{p['alpha_ci']:.2f}"
        ax.plot(n_dense, y_fit, "--", color=sty["color"], linewidth=1.3,
                alpha=0.7, zorder=3,
                label=f"{sty['label']} fit: $n^{{{alpha:.2f} \\pm {ci_str}}}$ ($R^2={p['r2']:.3f}$)")
        ax.fill_between(n_dense, y_lower, y_upper, color=sty["color"], alpha=0.10, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\mathrm{TTS}_{99}$")

    ax.set_xticks([10, 20, 30, 40, 50])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_exact_sci_notation))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs="auto"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    ax.legend(frameon=True, framealpha=0.9, edgecolor="black", loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
    return fig


DISEASE_STYLE = {
    "Diabetes Mellitus":    {"color": "#1f77b4", "marker": "o"},
    "Rheumatoid Arthritis": {"color": "#ff7f0e", "marker": "s"},
    "Asthma":               {"color": "#2ca02c", "marker": "^"},
    "Brain Neoplasms":      {"color": "#d62728", "marker": "D"},
}


def plot_tts_scaling_overlay(
    all_scaling,
    method,
    figsize=(6, 5),
    save_path=None,
    disease_style=None,
    show_title=True,
):
    """
    Overlay TTS scaling for all diseases for a single method ('sa' or 'sqa'), with polynomial fits.
    """
    if method not in ("sa", "sqa"):
        raise ValueError(f"method must be 'sa' or 'sqa', got {method!r}")
    if disease_style is None:
        disease_style = DISEASE_STYLE

    fig, ax = plt.subplots(figsize=figsize)
    n_dense = np.linspace(8, 55, 200)

    for dname, sr in all_scaling.items():
        if dname not in disease_style:
            continue
        a = analyze_scaling(sr)
        d = a[method]
        if len(d["sizes"]) == 0:
            continue
        sty = disease_style[dname]

        ax.errorbar(
            d["sizes"], d["tts"], yerr=d["tts_std"],
            fmt=sty["marker"], color=sty["color"], label=dname,
            markersize=7, capsize=4, elinewidth=1.5,
            markeredgecolor="black", markeredgewidth=0.8, zorder=5,
        )

        fits = d["fits"]
        if fits and "poly" in fits:
            y_fit = np.exp(poly_model(n_dense, *fits["poly"]["params"]))
            ax.plot(n_dense, y_fit, "--", color=sty["color"],
                    linewidth=2.5, alpha=0.6, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\mathrm{TTS}_{99}$")
    if show_title:
        ax.set_title(method.upper())

    ax.set_xticks([10, 20, 30, 40, 50])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_exact_sci_notation))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs="auto"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              frameon=True, framealpha=0.9, edgecolor="black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
    return fig
