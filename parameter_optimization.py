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

def create_qubo(
    z_base: np.ndarray,
    s_matrix: np.ndarray,
    gamma: float,
    beta: float,
    allowed_sizes: list[int] | None = None,
    penalty_strength: float | None = None
) -> dict[tuple[int,int], float]:
    """
    Build a QUBO with an optional constraint that the number of '1'-bits p
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

    return 1-ap, 1-weighted_rank_score


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
    
    best_ap_score = float('inf')
    best_ap_drug_ids = None
    best_ap_qubo = None
    best_ap_gamma = None
    best_ap_beta = None
    
    best_rank_score = float('inf')
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
                
                if ap < best_ap_score:
                    best_ap_score = ap
                    best_ap_drug_ids = list(drug_ids)
                    best_ap_qubo = qubo
                    best_ap_gamma = gamma
                    best_ap_beta = beta
                
                if rank < best_rank_score:
                    best_rank_score = rank
                    best_rank_drug_ids = list(drug_ids)
                    best_rank_qubo = qubo
                    best_rank_gamma = gamma
                    best_rank_beta = beta
        
    return (best_ap_drug_ids, 1-best_ap_score, best_ap_qubo, best_ap_gamma, best_ap_beta,
        best_rank_drug_ids, 1-best_rank_score, best_rank_qubo, best_rank_gamma, best_rank_beta,
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
    
    ap_min_idx = np.unravel_index(np.argmin(metric_matrices['ap']), metric_matrices['ap'].shape)
    ax1.scatter(
        beta_values[ap_min_idx[1]], 
        gamma_values[ap_min_idx[0]],
        color='red', s=100, label='Best Gaps'
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
    
    rank_min_idx = np.unravel_index(np.argmin(metric_matrices['rank']), metric_matrices['rank'].shape)
    ax2.scatter(
        beta_values[rank_min_idx[1]], 
        gamma_values[rank_min_idx[0]],
        color='red', s=100, label='Best Rank'
    )
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

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
    figsize: tuple = (12, 8)
):
    """
    Plot energy spectrum with color-coded match status.
    
    Args:
        sorted_results: List of (combination, energy, is_match)
        top_n: Number of solutions to plot
        disease_name: For plot title
        figsize: Figure dimensions
    """
    top_results = sorted_results[:top_n]
    energies = [res[1] for res in top_results]
    is_match = [res[2] for res in top_results]
    combinations = [res[0] for res in top_results]
    
    plt.figure(figsize=figsize)
    
    for i, (energy, match) in enumerate(zip(energies, is_match)):
        color = '#4CAF50' if match else '#F44336'  # Green for match, red for no match
        plt.hlines(y=i, xmin=min(energies)-0.1, xmax=energy, 
                   colors=color, linewidth=3, alpha=0.8)
        plt.scatter(energy, i, color=color, s=100, zorder=10)
        
        plt.text(energy, i, f" {energy:.4f}", va='center', fontsize=9, fontweight='bold')
        
        combo_str = ", ".join(sorted(combinations[i])) if combinations[i] else "[]"
        plt.text(min(energies)-0.11, i, combo_str, 
                 ha='right', va='center', fontsize=9, 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    plt.yticks(range(len(top_results)), [f"{i+1}" for i in range(len(top_results))])
    plt.xlabel('Energy', fontsize=12)
    plt.ylabel('Solution Rank', fontsize=12)
    plt.title(f'Top {len(top_results)} Solutions for {disease_name}', fontsize=14)
    
    energy_min = min(energies)
    energy_max = max(energies)
    plt.xlim(energy_min - 0.15 * (energy_max - energy_min), 
             energy_max + 0.05 * (energy_max - energy_min))
    
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Match with Dataset'),
        Patch(facecolor='#F44336', label='No Match')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

def plot_and_compute_precision_recall_curve(
    sorted_results: list[tuple[list[str], float, bool]],
    gt_combinations: set[frozenset],
    disease_name: str = "",
    figsize: tuple = (10, 7)
):
    """
    Plot precision-recall curve and compute average precision (AP). 
    """
    drug_set = set()
    for combo, _, _ in sorted_results:
        drug_set |= set(combo)
    relevant_gt = {c for c in gt_combinations if c.issubset(drug_set)}
    total_gt = len(relevant_gt)
    
    matches = np.array([match for _, _, match in sorted_results])
    
    cumulative_tp = np.cumsum(matches)
    cumulative_fp = np.cumsum(1 - matches)
    
    precision = cumulative_tp / (cumulative_tp + cumulative_fp)
    recall = cumulative_tp / total_gt if total_gt > 0 else np.zeros(len(matches))
    
    ap = 0.0
    prev_recall = 0.0
    for i in range(len(precision)):
        ap += precision[i] * (recall[i] - prev_recall)
        prev_recall = recall[i]
    
    plt.figure(figsize=figsize)
    
    plt.plot(recall, precision, 'b-', lw=2, label=f'Precision-Recall (AP = {ap:.3f})')
    
    positive_prop = np.mean(matches)
    plt.axhline(y=positive_prop, color='r', linestyle='--', label='Chance Level')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve: {disease_name}\n'
              f'({total_gt} Ground Truth Combinations)', fontsize=14)
    
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    
    plt.grid(alpha=0.2)
    plt.legend(loc='lower left', fontsize=10)
    
    plt.annotate(
        f'Average Precision: {ap:.3f}\n'
        f'Positives: {int(cumulative_tp[-1])}/{len(matches)}\n'
        f'Recall Range: {recall[0]:.3f}-{recall[-1]:.3f}',
        xy=(0.65, 0.25), xycoords='axes fraction',
        bbox=dict(boxstyle="round", fc="white", alpha=0.8)
    )
    
    plt.tight_layout()
    plt.show()
    
    return ap, precision, recall
