import numpy as np
import pandas as pd
from dataset_utils import get_ground_truth_combinations
from distance_metrics import analyze_disease_with_padding
import dimod
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from dimod.reference.samplers import ExactSolver
from dimod import BinaryQuadraticModel
from tqdm import tqdm
import os
import json
import gzip
from pathlib import Path
from typing import Dict, Tuple, Iterable, Union, List, Optional

QUBO = Dict[Tuple[int, int], float]

def create_qubo(
    z_base: np.ndarray,
    s_matrix: np.ndarray,
    gamma: float,
    beta: float,
    allowed_sizes: list[int] | None = None,
    penalty_strength: float | None = None
) -> dict[tuple[int,int], float]:
    """
    Build a QUBO with an optional constraint that the combination size
    lies in allowed_sizes (e.g. [2,3]).  If allowed_sizes is None or empty,
    no cardinality constraint is added.

    Args:
        z_base:  1D array of length M giving base linear terms.
        s_matrix: MxM symmetric matrix for quadratic couplings.
        gamma:   coefficient for the s_matrix term.
        beta:    bias added to each linear term.
        allowed_sizes: list of sizes p that are permitted (len ≤ 2).
        penalty_strength: strength A of the penalty.  If None, we set
            A = 10 * max(|z_base|, |gamma·s_matrix|) as a heuristic.
    """
    M = len(z_base)
    qubo = {}
    for i in range(M):
        qubo[(i, i)] = float(z_base[i] + beta)
    for i in range(M):
        for j in range(i+1, M):
            qubo[(i, j)] = float(-gamma * s_matrix[i, j])

    if not allowed_sizes:
        return qubo

    if len(allowed_sizes) > 2:
        raise ValueError("Can only constrain up to two allowed sizes with a quadratic penalty")

    if penalty_strength is None:
        max_lin = max(abs(v) for v in z_base + beta)
        max_quad = abs(gamma) * np.max(np.abs(s_matrix))
        A = 10 * max(max_lin, max_quad, 1.0)
    else:
        A = penalty_strength

    roots = allowed_sizes
    if len(roots) == 1:
        k = roots[0]
        a, b, c = 1.0, -2.0 * k, k * k
    else:
        k1, k2 = roots
        a, b, c = 1.0, -(k1 + k2), k1 * k2

    lin_shift = A * (a + b)
    quad_shift = A * (2 * a)

    for i in range(M):
        qubo[(i, i)] = qubo.get((i, i), 0.0) + lin_shift

    for i in range(M):
        for j in range(i+1, M):
            qubo[(i, j)] = qubo.get((i, j), 0.0) + quad_shift

    return qubo

def compute_qubo_metrics(
    qubo: dict,
    drug_ids: list[str],
    gt_combinations: set[frozenset],
    rank_power: float = 50
) -> tuple[float, float]:
    """
    Compute two exact metrics by full enumeration of all 2^n QUBO states:

    1. Average Precision (AP) treating each 2^n configuration as a scored instance,
       positives = ground truth combos, score = -energy.

    2. Power-law weighted rank metric:
         mean( ((M - rank(c) + 1)/M) ** rank_power )
       where rank(c)=1+#{E_all < E_c}, M=2^n.
    """
    n = len(drug_ids)
    drug_set = set(drug_ids)

    valid_gt = {c for c in gt_combinations if c.issubset(drug_set)}
    if not valid_gt:
        return 0.0, 0.0

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
    sampleset = ExactSolver().sample(bqm)

    all_energies = []
    all_matches = []
    for rec in sampleset.record:
        sel = {drug_ids[i] for i, bit in enumerate(rec.sample) if bit == 1}
        is_pos = frozenset(sel) in valid_gt
        all_energies.append(rec.energy)
        all_matches.append(is_pos)

    all_energies = np.array(all_energies)
    all_matches = np.array(all_matches, dtype=int)  # 1 for TP, 0 for FP

    M = len(all_energies)  # =2^n
    total_positives = all_matches.sum()

    order = np.argsort(all_energies)
    matches_sorted = all_matches[order]

    tp_cum = np.cumsum(matches_sorted)
    fp_cum = np.cumsum(1 - matches_sorted)

    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / total_positives

    ap = 0.0
    prev_r = 0.0
    for p_i, r_i in zip(precision, recall):
        ap += p_i * (r_i - prev_r)
        prev_r = r_i

    # Power‐law weighted rank metric
    gt_energies = all_energies[all_matches == 1]
    below_counts = np.array([
        np.sum(all_energies < E_c)
        for E_c in gt_energies
    ])
    ranks = 1 + below_counts
    linear_w = (M - ranks + 1) / M
    power_w = linear_w ** rank_power
    weighted_rank_score = power_w.mean()

    return ap, weighted_rank_score


def optimize_with_metrics(
    disease_name: str,
    df: pd.DataFrame,
    interactome_graph,
    filtered_disease_dict: dict,
    drug_to_targets: dict,
    dist_matrix: np.ndarray,
    nodes: list,
    node_to_idx: dict,
    num_qubits: int,
    gamma_values: list[float],
    beta_values: list[float],
    initial_drug_ids: list[str] = None,
    num_trials: int = 1,
    num_samples: int = 2000,
) -> tuple:
    """
    Optimize using both metrics.
    """
    if initial_drug_ids is None:
        initial_drug_ids = []

    gt_combinations = get_ground_truth_combinations(disease_name, df)
    
    best_ap_score = 0.0
    best_ap_drug_ids = None
    best_ap_qubo = None
    best_ap_gamma = None
    best_ap_beta = None
    
    best_rank_score = 0.0
    best_rank_drug_ids = None
    best_rank_qubo = None
    best_rank_gamma = None
    best_rank_beta = None
    
    metric_matrices = {
        'ap': np.full((len(gamma_values), len(beta_values)), float('inf')),
        'rank': np.full((len(gamma_values), len(beta_values)), float('inf'))
    }
    
    for trial in range(num_trials):
        _, z_dict, s_matrix, drug_ids = analyze_disease_with_padding(
            disease_name, df, interactome_graph, filtered_disease_dict,
            drug_to_targets, dist_matrix, nodes, node_to_idx, initial_drug_ids,
            num_qubits, num_samples=num_samples
        )
        print(f"Selected drugs: {drug_ids}")
        z_base = np.array([z_dict[did] for did in drug_ids])
        
        for i, gamma in enumerate(gamma_values):
            for j, beta in enumerate(beta_values):
                qubo = create_qubo(z_base, s_matrix, gamma, beta, [2,3])
                
                ap, rank = compute_qubo_metrics(
                    qubo, drug_ids, gt_combinations)
                
                metric_matrices['ap'][i, j] = ap
                metric_matrices['rank'][i, j] = rank
                
                if ap > best_ap_score:
                    best_ap_score = ap
                    best_ap_drug_ids = list(drug_ids)
                    best_ap_qubo = qubo
                    best_ap_gamma = gamma
                    best_ap_beta = beta
                
                if rank > best_rank_score:
                    best_rank_score = rank
                    best_rank_drug_ids = list(drug_ids)
                    best_rank_qubo = qubo
                    best_rank_gamma = gamma
                    best_rank_beta = beta
        
    return (best_ap_drug_ids, best_ap_score, best_ap_qubo, best_ap_gamma, best_ap_beta,
        best_rank_drug_ids, best_rank_score, best_rank_qubo, best_rank_gamma, best_rank_beta,
        metric_matrices)

def plot_metric_results(metric_matrices, gamma_values, beta_values, disease_name):
    """Plot both metrics side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    ap_plot = ax1.imshow(
        metric_matrices['ap'],
        cmap='viridis',  
        aspect='auto',
        origin='lower',
        extent=[beta_values[0], beta_values[-1], gamma_values[0], gamma_values[-1]]
    )
    ax1.set_title(f'Average Precision: {disease_name}')
    ax1.set_xlabel('Beta')
    ax1.set_ylabel('Gamma')
    fig.colorbar(ap_plot, ax=ax1, label='AP Score')
    
    ap_max_idx = np.unravel_index(np.argmax(metric_matrices['ap']), metric_matrices['ap'].shape)
    ax1.scatter(
        beta_values[ap_max_idx[1]], 
        gamma_values[ap_max_idx[0]],
        color='red', s=100, label='Best AP'
    )
    ax1.legend()
    
    rank_plot = ax2.imshow(
        metric_matrices['rank'],
        cmap='viridis', 
        aspect='auto',
        origin='lower',
        extent=[beta_values[0], beta_values[-1], gamma_values[0], gamma_values[-1]]
    )
    ax2.set_title(f'Weighted Rank Metric: {disease_name}')
    ax2.set_xlabel('Beta')
    ax2.set_ylabel('Gamma')
    fig.colorbar(rank_plot, ax=ax2, label='W-Rank Score')
    
    rank_max_idx = np.unravel_index(np.argmax(metric_matrices['rank']), metric_matrices['rank'].shape)
    ax2.scatter(
        beta_values[rank_max_idx[1]], 
        gamma_values[rank_max_idx[0]],
        color='red', s=100, label='Best Rank'
    )
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def optimize_for_ap(
    disease_name: str,
    df: "pd.DataFrame",
    interactome_graph,
    filtered_disease_dict: dict,
    drug_to_targets: dict,
    dist_matrix: np.ndarray,
    nodes: list,
    node_to_idx: dict,
    num_qubits: int,
    gamma_values: List[float],
    beta_values: List[float],
    initial_drug_ids: Optional[List[str]] = None,
    num_trials: int = 1,
    num_samples: int = 2000,
) -> Tuple[
    Optional[List[str]],
    float,
    Optional[np.ndarray],
    Optional[float],
    Optional[float],
    dict,
]:
    """
    Optimize for average-precision (AP) metric over gamma/beta grid.
    """
    if initial_drug_ids is None:
        initial_drug_ids = []

    gt_combinations = get_ground_truth_combinations(disease_name, df)

    best_ap_score = 0.0
    best_ap_drug_ids = None
    best_ap_qubo = None
    best_ap_gamma = None
    best_ap_beta = None

    ap_matrix = np.full((len(gamma_values), len(beta_values)), np.nan, dtype=float)
    metric_matrices = {'ap': ap_matrix}

    for trial in range(num_trials):
        _, z_dict, s_matrix, drug_ids = analyze_disease_with_padding(
            disease_name, df, interactome_graph, filtered_disease_dict,
            drug_to_targets, dist_matrix, nodes, node_to_idx, initial_drug_ids,
            num_qubits, num_samples=num_samples
        )
        print(f"Selected drugs: {drug_ids}")
        z_base = np.array([z_dict[did] for did in drug_ids])

        for i, gamma in enumerate(gamma_values):
            for j, beta in enumerate(beta_values):
                qubo = create_qubo(z_base, s_matrix, gamma, beta, [2, 3])

                ap, _ = compute_qubo_metrics(qubo, drug_ids, gt_combinations)

                metric_matrices['ap'][i, j] = float(ap)

                if ap > best_ap_score:
                    best_ap_score = float(ap)
                    best_ap_drug_ids = list(drug_ids)
                    best_ap_qubo = qubo
                    best_ap_gamma = gamma
                    best_ap_beta = beta

    fig, ax = plot_ap_landscape(metric_matrices, gamma_values, beta_values, disease_name=disease_name)

    return (
        best_ap_drug_ids,
        best_ap_score,
        best_ap_qubo,
        best_ap_gamma,
        best_ap_beta,
        metric_matrices,
        fig,
        ax,
    )

def plot_ap_landscape(
    metric_matrices: dict,
    gamma_values: list,
    beta_values: list,
    disease_name: str = "",
    fig_size: tuple = (7.5, 6),
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0
) -> tuple:
    def _edges_from_centers(centers: np.ndarray) -> np.ndarray:
        centers = np.asarray(centers, dtype=float)
        if centers.size == 1:
            d = 1.0
            return np.array([centers[0] - d / 2.0, centers[0] + d / 2.0])
        mids = 0.5 * (centers[:-1] + centers[1:])
        left = centers[0] - 0.5 * (centers[1] - centers[0])
        right = centers[-1] + 0.5 * (centers[-1] - centers[-2])
        return np.concatenate(([left], mids, [right]))

    ap_grid = np.asarray(metric_matrices['ap'], dtype=float)

    beta_edges = _edges_from_centers(np.asarray(beta_values))
    gamma_edges = _edges_from_centers(np.asarray(gamma_values))

    fig, ax = plt.subplots(figsize=fig_size)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 16,  
        "legend.fontsize": 10,
        "xtick.labelsize": 14,  
        "ytick.labelsize": 14,
        "figure.dpi": 150,
        "svg.fonttype": "none"  
    })

    try:
        from scipy.ndimage import gaussian_filter
        ap_plot_grid = gaussian_filter(ap_grid, sigma=1.0)
    except Exception:
        ap_plot_grid = ap_grid.copy()

    pcm = ax.pcolormesh(
        beta_edges,
        gamma_edges,
        ap_plot_grid,
        shading='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    cb = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)

    cb.set_label('')
    cb.ax.tick_params(labelsize=14)

    b_centers, g_centers = np.meshgrid(beta_values, gamma_values)
    contour_levels = np.linspace(np.nanmax([vmin, ap_plot_grid.min()]), np.nanmin([vmax, ap_plot_grid.max()]), 6)
    try:
        cs = ax.contour(
            b_centers,
            g_centers,
            ap_plot_grid,
            levels=contour_levels,
            colors='white',
            linewidths=0.8,
            alpha=0.6
        )
        ax.clabel(cs, fmt="%.2f", fontsize=12, colors='white')
    except Exception:
        pass

    best_idx = np.nanargmax(ap_grid)
    best_i, best_j = np.unravel_index(best_idx, ap_grid.shape)
    best_gamma = gamma_values[best_i]
    best_beta = beta_values[best_j]
    best_ap = float(ap_grid[best_i, best_j])
    buffer = 0.03 * (np.max(gamma_edges) - np.min(gamma_edges))
    ax.scatter([best_beta], [best_gamma], color='red', s=80, edgecolor='black', linewidth=0.8, zorder=10)

    ax.set_xlabel('$\\beta$', fontsize=16)
    ax.set_ylabel('$\\gamma$', fontsize=16)

    ax.set_xlim(beta_edges[0], beta_edges[-1])
    ax.set_ylim(gamma_edges[0], gamma_edges[-1])
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.set_aspect('auto')
    ax.grid(False)

    plt.tight_layout()
    return fig, ax


def get_exact_low_energy_states(
    qubo: dict,
    drug_ids: list[str],
    gt_combinations: set[frozenset],
    max_states: int = 5000,  # Controls maximum states
    energy_threshold: float = 5.0  # Controls energy range
) -> list[tuple[list[str], float, bool]]:
    """
    Find the exact lowest energy states.

    Returns:
        List of (drug_combination, energy, is_match) sorted by energy
    """
    n = len(drug_ids)
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

    # For small problems, we can use brute-force enumeration
    sampler = dimod.ExactSolver()
    sampleset = sampler.sample(bqm)
    all_samples = list(sampleset.data())
    all_samples.sort(key=lambda s: s.energy)
        
    ground_energy = all_samples[0].energy
        
    filtered_samples = [s for s in all_samples if s.energy <= ground_energy + energy_threshold]
        
    return _process_samples(filtered_samples[:max_states], drug_ids, gt_combinations)
    
def _process_samples(samples, drug_ids, gt_combinations):
    """Convert samples to results format"""
    results = []
    seen = set()
    for sample in samples:
        state = tuple(sample.sample.values())
        if state in seen:
            continue
        seen.add(state)
        
        selected_drugs = [drug_ids[i] for i, val in enumerate(state) if val == 1]
        is_match = frozenset(selected_drugs) in gt_combinations
        results.append((selected_drugs, float(sample.energy), is_match))
    return results

def plot_energy_spectrum(
    sorted_results: list[tuple[list[str], float, bool]],
    top_n: int = 20,
    disease_name: str = "",
    figsize: tuple = (12, 10)
):
    """
    Plot energy spectrum with color-coded match status.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 20,          
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "legend.fontsize": 15,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "figure.dpi": 300
    })

    top_n = min(top_n, len(sorted_results))
    top_results = sorted_results[:top_n]
    energies = [res[1] for res in top_results]
    is_match = [res[2] for res in top_results]
    combinations = [res[0] for res in top_results]

    fig, ax = plt.subplots(figsize=figsize)

    energy_min = min(energies) if energies else 0.0
    energy_max = max(energies) if energies else 1.0
    energy_range = max(1e-8, (energy_max - energy_min))

    pad_left = 0.18 * energy_range
    pad_right = 0.12 * energy_range   
    x_lim_left = energy_min - pad_left
    x_lim_right = energy_max + pad_right

    x_line_start = x_lim_left + 0.03 * energy_range

    for i, (energy, match) in enumerate(zip(energies, is_match)):
        color = '#2ca02c' if match else '#d62728'
        ax.hlines(y=i, xmin=x_line_start, xmax=energy,
                  colors=color, linewidth=3.0, alpha=0.95, zorder=2)
        ax.scatter(energy, i, color=color, s=150, zorder=4,
                   edgecolor='black', linewidth=0.6)
        '''
        dx = 0.015 * energy_range
        text_margin = 0.008 * energy_range
        if energy + dx + text_margin > (x_lim_right - 0.02 * energy_range):
            text_x = energy - dx
            ha = 'right'
        else:
            text_x = energy + dx
            ha = 'left'

        ax.text(text_x, i, f"{energy:.2f}", ha=ha, va='center',
                fontsize=15, fontweight='bold', zorder=6,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=0.6),
                clip_on=False)
        '''
    ax.set_yticks(range(len(top_results)))
    ax.set_yticklabels([f"{i+1}" for i in range(len(top_results))], fontsize=21)
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    ax.tick_params(axis='y', labelleft=True, labelright=False)

    ax.set_xlabel('Energy', fontsize=28)
    ax.set_ylabel('Solution rank', fontsize=28)

    ax.set_xlim(x_lim_left, x_lim_right)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markeredgecolor='k', markersize=12, label='Match with dataset'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
               markeredgecolor='k', markersize=12, label='No match')
    ]
    leg = ax.legend(handles=legend_handles, loc='lower right', frameon=True, fontsize=20)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_alpha(0.95)

    ax.grid(axis='x', linestyle='--', alpha=0.35)

    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()
    return fig


def plot_and_compute_precision_recall_curve(
    sorted_results,
    gt_combinations,
    disease_name="",
    figsize=(5.5, 4.5),
    dpi=300,
    annotate_ap_inplot=True,
    draw_step=False
):
    """ 
    Plot precision-recall curve and compute average precision (AP). 
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": dpi
    })

    drug_set = set()
    for combo, _, _ in sorted_results:
        drug_set |= set(combo)
    relevant_gt = {c for c in gt_combinations if c.issubset(drug_set)}
    total_gt = len(relevant_gt)

    matches = np.array([match for _, _, match in sorted_results], dtype=int)

    cumulative_tp = np.cumsum(matches)
    cumulative_fp = np.cumsum(1 - matches)
    denom = (cumulative_tp + cumulative_fp).astype(float)
    precision = np.divide(cumulative_tp, denom, out=np.zeros_like(denom, dtype=float), where=denom != 0)
    total_pos_in_results = int(cumulative_tp[-1]) if len(cumulative_tp) > 0 else 0
    recall = cumulative_tp / float(total_pos_in_results) if total_pos_in_results > 0 else np.zeros_like(precision)

    print(f"Total relevant ground truth combinations: {total_gt}")

    ap = 0.0
    prev_r = 0.0
    for p_i, r_i in zip(precision, recall):
        ap += float(p_i) * (r_i - prev_r)
        prev_r = r_i

    fig, ax = plt.subplots(figsize=figsize)
    if draw_step:
        ax.plot(recall, precision, drawstyle='steps-post', lw=2.2, zorder=3)
    else:
        ax.plot(recall, precision, '-', lw=2.2, zorder=3)

    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_xlim(0.0, 1.02)   
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.18, linestyle='--', linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if annotate_ap_inplot:
        ax.text(
            0.95, 0.95, f'AP = {ap:.3f}',
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
        )

    plt.tight_layout()
    return ap, precision, recall, fig


def save_qubo_to_file(qubo: QUBO, filename: Union[str, Path], compress: bool = False) -> None:
    """
    Saves a QUBO dict[(i,j), float] to a JSON file.
    """
    filename = Path(filename)
    if qubo:
        max_idx = max(max(i, j) for (i, j) in qubo.keys())
        M = int(max_idx) + 1
    else:
        M = 0

    entries = [[int(i), int(j), float(v)] for (i, j), v in qubo.items()]

    payload = {"M": M, "entries": entries}

    do_gzip = compress or filename.suffix == ".gz"

    if do_gzip:
        with gzip.open(filename, "wt", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    else:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def load_qubo_from_file(filename: Union[str, Path]) -> QUBO:
    """
    Reads a QUBO saved by `save_qubo_to_file` and returns dict[(i,j), float].
    """
    filename = Path(filename)
    do_gzip = filename.suffix == ".gz"

    if do_gzip:
        with gzip.open(filename, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        with open(filename, "r", encoding="utf-8") as f:
            payload = json.load(f)

    if not isinstance(payload, dict) or "entries" not in payload:
        raise ValueError("File does not contain expected qubo payload (missing 'entries').")

    entries = payload["entries"]
    qubo: QUBO = {}
    for triple in entries:
        if not (isinstance(triple, list) or isinstance(triple, tuple)) or len(triple) != 3:
            raise ValueError("Each entry must be a 3-element list [i, j, value].")
        i, j, v = triple
        qubo[(int(i), int(j))] = float(v)

    return qubo
