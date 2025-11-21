import numpy as np
from collections import Counter
from typing import Dict, Tuple, List, Set, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import csv
import itertools
import pandas as pd
from dwave.samplers import PathIntegralAnnealingSampler
import dimod
from qubo_selection import get_sorted_results_allowed_sizes

def make_exponential_beta_schedule(beta_min: float, beta_max: float, n_steps: int, alpha: float):
    if n_steps < 2:
        return [float(beta_min), float(beta_max)]
    t = np.linspace(0.0, 1.0, n_steps)
    if alpha == 0.0:
        norm = t
    else:
        num = np.expm1(alpha * t)
        den = np.expm1(alpha)
        if den == 0.0:
            norm = t
        else:
            norm = num / den
    betas = beta_min + (beta_max - beta_min) * norm
    return [float(x) for x in betas.tolist()]

def plot_frequency_by_rank(
    counts: Counter,
    sorted_results: List[Tuple[List[str], float, bool]],
    num_reads: int,
    top_k: int = 10,
    figsize: Tuple[float, float] = (8.0, 3.5),
    normalize: bool = True,
    match_color: str = "#1b9e77",
    nonmatch_color: str = "#d9d9d9",
    label_fontsize: int = 20,
    tick_fontsize: int = 16,
):
    rank_lookup = {}
    for rank, (sel_list, energy, is_match) in enumerate(sorted_results, start=1):
        key = tuple(sorted(sel_list))
        rank_lookup[key] = (rank, energy, bool(is_match))

    max_rank_available = len(sorted_results)
    K = min(top_k, max_rank_available)
    ranks = list(range(1, K+1))
    freqs = []
    colors = []
    labels = []
    for r in ranks:
        sel_list, energy, is_match = sorted_results[r-1]
        key = tuple(sorted(sel_list))
        count = int(counts.get(key, 0))
        val = (count / float(num_reads)) if normalize else count
        freqs.append(val)
        colors.append(match_color if is_match else nonmatch_color)
        labels.append((key, energy, is_match))

    plt_rc = {
        "font.family": "sans-serif",
        "font.size": label_fontsize,
        "axes.titlesize": label_fontsize,
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
    }
    plt.rcParams.update(plt_rc)

    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.85
    x = np.arange(len(ranks))
    bars = ax.bar(x, freqs, width=bar_width, color=colors, edgecolor="none")

    ax.set_xlabel("Solution rank", fontsize=label_fontsize)
    ylabel = "Frequency" if normalize else f"Occurrences (out of {num_reads})"
    ax.set_ylabel(ylabel, fontsize=label_fontsize)

    xtick_labels = []
    for (_, energy, _) in [labels[i] for i in range(len(labels))]:
        xtick_labels.append(f"{ranks[len(xtick_labels)]}")
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=tick_fontsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.margins(x=0)
    ax.set_xlim(-0.5, len(ranks)-0.5)

    legend_handles = [
        Patch(facecolor=match_color, label="Match"),
        Patch(facecolor=nonmatch_color, label="Prediction"),
    ]
    ax.legend(handles=legend_handles, frameon=True, fontsize=tick_fontsize, framealpha=1.0 )

    for rect, (combo, energy, is_match) in zip(bars, labels):
        height = rect.get_height()
        if normalize:
            txt = f"{height:.3f}"
        else:
            txt = f"{int(height)}"
        ax.annotate(txt,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=tick_fontsize)

    plt.tight_layout()
    return fig, ax

def run_dwave_sqa(
    qubo: Dict[Tuple[int,int], float],
    drug_ids: List[str],
    *,
    beta_range: Optional[Tuple[float, float]] = None,
    beta_schedule_type: str = "geometric",
    beta_schedule: Optional[List[float]] = None,
    initial_states: Optional[List[Dict[int,int]]] = None,
    initial_states_generator: Optional[str] = None,
    interrupt_function: Optional[Any] = None,
    num_reads: int = 512,
    num_sweeps: Optional[int] = None,
    num_sweeps_per_beta: int = 1,
    seed: Optional[int] = None,
    Gamma: float = 1.0,
    Hp_field: Optional[List[float]] = None,
    Hd_field: Optional[List[float]] = None,
    **extra_kwargs
) -> Dict[str, Any]:

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
    sampler = PathIntegralAnnealingSampler()

    kwargs = {}
    if beta_range is not None:
        kwargs["beta_range"] = beta_range
    kwargs["beta_schedule_type"] = beta_schedule_type
    if initial_states is not None:
        kwargs["initial_states"] = initial_states
    if initial_states_generator is not None:
        kwargs["initial_states_generator"] = initial_states_generator
    if interrupt_function is not None:
        kwargs["interrupt_function"] = interrupt_function
    if num_reads is not None:
        kwargs["num_reads"] = num_reads
    if num_sweeps is not None:
        kwargs["num_sweeps"] = num_sweeps
    kwargs["num_sweeps_per_beta"] = num_sweeps_per_beta
    if seed is not None:
        kwargs["seed"] = seed
    if Hp_field is not None:
        kwargs["Hp_field"] = list(Hp_field)
    if Hd_field is not None:
        kwargs["Hd_field"] = list(Hd_field)
    kwargs["Gamma"] = Gamma
    kwargs.update(extra_kwargs)

    response = sampler.sample(bqm, **kwargs)

    vars_order = list(response.variables)

    rec = response.record
    samples_matrix = np.array(rec['sample'])
    energies = [float(e) for e in rec['energy']]

    n_reads, n_vars = samples_matrix.shape

    combos = []
    for r in range(n_reads):
        row = samples_matrix[r]
        x = [0] * len(drug_ids)
        for col_idx, var_label in enumerate(vars_order):
            bit = int(row[col_idx])
            drug_idx = var_label
            x[drug_idx] = bit
        sel = tuple(sorted([drug_ids[i] for i, bit in enumerate(x) if bit == 1]))
        combos.append(sel)

    counts = Counter(combos)
    sample_energies = list(zip(combos, energies))
    return {"counts": counts, "num_reads": n_reads, "sample_energies": sample_energies, "response": response}

def summarize_and_save_topk(counts: Counter, sorted_results: List[Tuple[List[str], float, bool]], num_reads: int, out_csv_path: str, top_k: int = 15):
    rows = []
    for rank, (sel_list, energy, is_match) in enumerate(sorted_results[:top_k], start=1):
        key = tuple(sorted(sel_list))
        cnt = int(counts.get(key, 0))
        frac = cnt / float(num_reads) if num_reads > 0 else 0.0
        rows.append({"rank": rank, "combo": "|".join(key), "energy": energy, "is_match": is_match, "count": cnt, "fraction": frac})

    rank1_count = rows[0]["count"] if rows else 0
    frac_on_rank1 = rank1_count / float(num_reads) if num_reads > 0 else 0.0
    frac_in_top5 = sum(r["count"] for r in rows[:5]) / float(num_reads) if num_reads > 0 else 0.0
    summary = {"frac_on_rank1": frac_on_rank1, "frac_in_top5": frac_in_top5, "num_reads": num_reads}

    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)

    return summary

def parameter_exploration(
    qubo: Dict[Tuple[int,int], float],
    drug_ids: List[str],
    gt_combinations: Set[frozenset],
    outdir: str = "sqa_plots",
    top_k: int = 10,
    num_reads_default: int = 2048,
    repeats_per_setting: int = 3,
):
    os.makedirs(outdir, exist_ok=True)

    sorted_results = get_sorted_results_allowed_sizes(qubo, drug_ids, gt_combinations, allowed_sizes=[2,3])

    beta_low = 0.01
    base_num_reads = num_reads_default

    grid = []
    for bh in [1.9]:
        for sched in ["linear"]:
            for ns in [1]:
                for spb in [2]:
                    for gm in [1.0]:
                        grid.append({"beta_range": (beta_low, bh), "beta_schedule_type": sched, "num_sweeps": ns, "num_sweeps_per_beta": spb, "Gamma": gm, "num_reads": base_num_reads})

    custom_params_space = []
    for bh in [3.0]:
        for n_steps in [20001]: 
            for alpha in [1.5]: 
                for gm in [1.0]:
                    custom_params_space.append({"beta_min": beta_low, "beta_max": bh, "n_steps": n_steps, "alpha": alpha, "Gamma": gm, "num_reads": base_num_reads, "num_sweeps": None, "num_sweeps_per_beta": 2})

    for cp in custom_params_space:
        Hp_field = make_exponential_beta_schedule(0.0, cp["beta_max"], cp["n_steps"], cp["alpha"])
        Hd_field = Hp_field[::-1]
        grid.append({
            "beta_schedule_type": "custom",
            "Hp_field": Hp_field,
            "Hd_field": Hd_field,
            "num_sweeps": cp["num_sweeps"],
            "num_sweeps_per_beta": cp["num_sweeps_per_beta"],
            "Gamma": cp["Gamma"],
            "num_reads": cp["num_reads"],
            "alpha": cp["alpha"],
            "n_steps": cp["n_steps"],
            "beta_max": cp["beta_max"],})
        
    def _hashable(x):
        if x is None:
            return None
        if isinstance(x, list):
            try:
                return tuple(x)
            except Exception:
                return str(x)
        return x

    def key_from_params(p):
        return (
            _hashable(p.get("beta_range")),
            p.get("beta_schedule_type"),
            p.get("num_sweeps"),
            _hashable(p.get("Hp_field")),
            _hashable(p.get("Hd_field")),
            p.get("alpha"),
            p.get("num_sweeps_per_beta"),
            p.get("Gamma"),
            p.get("num_reads"),
        )

    seen = set()
    uniq_grid = []
    for p in grid:
        k = key_from_params(p)
        if k not in seen:
            uniq_grid.append(p); seen.add(k)
    grid = uniq_grid

    print(f"Running {len(grid)} unique parameter settings, each repeated {repeats_per_setting} times -> total runs = {len(grid)*repeats_per_setting}")

    run_index = 0
    for params in grid:
        for repeat in range(repeats_per_setting):
            run_index += 1
            seed = 1000 + 10*repeat
            print(f"[{run_index}] params: {params.get('beta_schedule_type')} repeat {repeat+1}/{repeats_per_setting} seed={seed}")

            Hp_field = params.get("Hp_field")
            Hd_field = params.get("Hd_field")

            if params.get("beta_schedule_type") == "custom":
                num_sweeps = None
            else:
                num_sweeps = int(params.get("num_sweeps"))

            try:
                out = run_dwave_sqa(
                    qubo,
                    drug_ids,
                    beta_range=params.get("beta_range"),
                    beta_schedule_type=params.get("beta_schedule_type"),
                    num_reads=int(params.get("num_reads")),
                    num_sweeps=num_sweeps,
                    num_sweeps_per_beta=int(params.get("num_sweeps_per_beta")),
                    seed=seed,
                    Gamma=float(params.get("Gamma")),
                    Hp_field=Hp_field,
                    Hd_field=Hd_field,
                )

            except Exception as ex:
                print("Sampler failed for params:", params, "error:", ex)
                continue

            counts = out["counts"]
            nreads = out["num_reads"]

            sched_tag = params.get("beta_schedule_type")
            if sched_tag == "custom":
                n_steps_val = params.get('n_steps') if params.get('n_steps') is not None else len(params.get('Hp_field', []))
                alpha_val = params.get('alpha', 'N/A')
                beta_max_val = params.get('beta_max', (params.get('beta_range') or (None, None))[1])
                png_name = f"rankfreq_custom_n{n_steps_val}_alpha{alpha_val}_bmax{beta_max_val}_reads{nreads}_r{repeat+1}"
            else:
                png_name = f"rankfreq_br{params.get('beta_range')[1]}_{params.get('beta_schedule_type')}_sweeps{params.get('num_sweeps')}_spb{params.get('num_sweeps_per_beta')}_G{params.get('Gamma')}_r{repeat+1}"

            png_path = os.path.join(outdir, png_name + ".pdf")
            fig, ax = plot_frequency_by_rank(counts, sorted_results, normalize=True, num_reads=nreads, top_k=top_k, figsize=(10,4))
            fig.savefig(png_path, format='pdf', bbox_inches="tight")
            plt.close(fig)

            csv_path = os.path.join(outdir, png_name + ".csv")
            summary = summarize_and_save_topk(counts, sorted_results, nreads, csv_path, top_k=top_k)
            
            print(f"run={run_index} type={'custom' if params.get('Hp_field') else 'builtin'} alpha={params.get('alpha','N/A')} n_steps={(params.get('n_steps') if params.get('n_steps') is not None else (len(params.get('Hp_field') or []) if params.get('Hp_field') is not None else 'N/A'))} beta_max={params.get('beta_max', (params.get('beta_range') or (None, None))[1])} spb={params.get('num_sweeps_per_beta')} reads={params.get('num_reads')} nsweeps={params.get('num_sweeps')} Gamma={params.get('Gamma')}")

            print(f"Saved {png_path} | top1_frac={summary['frac_on_rank1']:.4f} top5_frac={summary['frac_in_top5']:.4f}")

    print("All parameter runs complete. Check:", outdir)
