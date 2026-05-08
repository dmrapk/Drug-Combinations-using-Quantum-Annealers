from dataset_utils import get_ground_truth_combinations
from typing import List, Tuple, Dict, Any, Optional, Set, FrozenSet, Any
import dimod
import numpy as np
import itertools
import pandas as pd
import os
from dimod import SimulatedAnnealingSampler
from distance_metrics import analyze_disease_with_padding
from parameter_optimization import create_qubo
from math import comb
import matplotlib.pyplot as plt
from parameter_optimization import load_qubo_from_file

def find_best_qubo_for_params_ap_enumeration(
    disease_name: str,
    df,
    interactome_graph,
    filtered_disease_dict: dict,
    drug_to_targets: dict,
    dist_matrix: np.ndarray,
    nodes: list,
    node_to_idx: dict,
    initial_drug_ids: List[str],
    num_qubits: int,
    gamma: float,
    beta: float,
    num_trials: int = 20,
    num_samples_for_z: int = 1000,
    allowed_sizes: Optional[List[int]] = [2,3],
    return_curve: bool = False
) -> Tuple[List[str], Dict[Tuple[int,int], float], float, List[Dict[str, Any]]]:
    """
    Runs multiple trials to find the best QUBO for given gamma and beta parameters,
    optimizing for average precision (AP) using enumeration for the restricted combinations constrained by allowed_sizes.
    """

    gt_combinations = get_ground_truth_combinations(disease_name, df)

    best_ap = -np.inf
    best_drug_ids = []
    best_qubo = {}
    trials_info = []
    successful = 0

    for t in range(num_trials):
        try:
            _, z_dict, s_matrix, drug_ids = analyze_disease_with_padding(
                disease_name=disease_name, df=df, interactome_graph=interactome_graph,
                filtered_disease_dict=filtered_disease_dict, drug_to_targets=drug_to_targets,
                dist_matrix=dist_matrix, nodes=nodes, node_to_idx=node_to_idx,
                initial_drug_ids=initial_drug_ids, num_qubits=num_qubits,
                num_samples=num_samples_for_z
            )
        except Exception as e:
            print(f"[Warn] Trial {t}: Z-score setup failed: {e}")

        z_base = np.array([z_dict[did] for did in drug_ids])
        try:
            qubo = create_qubo(z_base, s_matrix, gamma, beta, allowed_sizes)
        except Exception as e:
            print(f"[Warn] Trial {t}: create_qubo failed: {e}")

        n = len(drug_ids)
        total_candidates = 0
        for k in allowed_sizes:
            if 0 <= k <= n:
                total_candidates += comb(n, k)
        # safety cap warning is set to 1e6 candidate combos
        if total_candidates > 1000000:
            print(f"[INFO] enumeration candidate count {total_candidates} too large")

        try:
            ap = compute_average_precision_allowed_sizes(
                qubo, drug_ids, gt_combinations, allowed_sizes=allowed_sizes,return_curve=return_curve)
            ap = float(ap)
        except Exception as e:
            print(f"[Warn] Trial {t}: AP computation failed: {e}")
            continue

        successful += 1
        trials_info.append({'trial': t, 'drug_ids': drug_ids.copy(), 'ap': ap})
        print(f"[Trial {t}] ap={ap:.6f}")

        if ap > best_ap:
            best_ap = ap
            best_drug_ids = drug_ids.copy()
            best_qubo = qubo.copy()

    if successful == 0:
        raise RuntimeError("No successful trials; cannot pick best QUBO.")

    return best_drug_ids, best_qubo, best_ap, trials_info

def compute_average_precision_allowed_sizes(
    qubo: Dict[Tuple[int,int], float],
    drug_ids: List[str],
    gt_combinations: Set[frozenset],
    allowed_sizes: List[int] = [2,3],
    return_curve: bool = False
) -> Any:
    """
    Computes the exact Average Precision by enumerating only combinations with sizes in allowed_sizes for feasability instead of all 2^n combinations.

    Returns ap (float) or (ap, precision, recall, scores) if return_curve=True.
    """
    n = len(drug_ids)
    drug_set = set(drug_ids)
    valid_gt = {c for c in gt_combinations if c.issubset(drug_set)}
    if not valid_gt:
        if return_curve:
            return 0.0, np.array([]), np.array([]), np.array([])
        return 0.0

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

    candidates = []
    for k in allowed_sizes:
        if k < 0 or k > n:
            continue
        candidates.extend(itertools.combinations(range(n), k))

    if len(candidates) == 0:
        if return_curve:
            return 0.0, np.array([]), np.array([]), np.array([])
        return 0.0

    energies = []
    matches = []
    scores = []

    for comb in candidates:
        sample = {i: (1 if i in comb else 0) for i in range(n)}
        E = float(bqm.energy(sample))
        energies.append(E)
        sel = frozenset(drug_ids[i] for i in comb)
        matches.append(1 if sel in valid_gt else 0)
        scores.append(-E)   

    energies = np.array(energies)
    matches = np.array(matches, dtype=int)
    scores = np.array(scores)

    order = np.argsort(-scores)
    matches_sorted = matches[order]

    tp = np.cumsum(matches_sorted)
    fp = np.cumsum(1 - matches_sorted)
    denom = tp + fp
    precision = np.divide(tp, denom, out=np.zeros_like(tp, dtype=float), where=denom>0)
    total_pos = int(tp[-1]) if tp.size else 0
    if total_pos == 0:
        if return_curve:
            return 0.0, precision, np.zeros_like(precision), scores[order]
        return 0.0
    recall = tp / float(total_pos)
    ap = 0.0
    prev_r = 0.0
    for p_i, r_i in zip(precision, recall):
        ap += float(p_i) * (r_i - prev_r)
        prev_r = r_i

    print(f"Enumerated {len(candidates)} candidates; positives={total_pos}; AP={ap:.6f}")
    # print top 10
    for idx in range(min(10, len(order))):
        pos = order[idx]
        comb_idx = candidates[pos]
        sel = tuple(drug_ids[i] for i in comb_idx)
        print(f" #{idx+1}: energy={energies[pos]:.6g}, pos={matches[pos]}, sel={sel}")

    if return_curve:
        plot_precision_recall_curve_from_pr(precision, recall, ap)
    return float(ap)

def plot_precision_recall_curve_from_pr(
    precision: np.ndarray,
    recall: np.ndarray,
    ap: float,
    figsize: Tuple[int, int] = (12, 6)
):
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, lw=2, label=f'Precision-Recall (AP = {ap:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve:', fontsize=14)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(alpha=0.2)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

def get_sorted_results_allowed_sizes(
    qubo: Dict[Tuple[int,int], float],
    drug_ids: List[str],
    gt_combinations: Set[frozenset],
    allowed_sizes: List[int] = [2, 3]
) -> List[Tuple[List[str], float, bool]]:
    """
    Enumerate all index-combinations of sizes in allowed_sizes, compute energies,
    and return a list sorted by ascending energy:
    """
    n = len(drug_ids)
    drug_set = set(drug_ids)
    valid_gt = {c for c in gt_combinations if c.issubset(drug_set)}
    if not valid_gt:
        raise ValueError("No ground-truth combinations valid for provided drug_ids")

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

    candidates = []
    for k in allowed_sizes:
        if 0 <= k <= n:
            candidates.extend(itertools.combinations(range(n), k))

    results = []
    for comb in candidates:
        sample = {i: (1 if i in comb else 0) for i in range(n)}
        energy = float(bqm.energy(sample))
        sel = tuple(sorted(drug_ids[i] for i in comb))
        is_match = frozenset(sel) in valid_gt
        results.append((list(sel), energy, bool(is_match)))

    results.sort(key=lambda t: t[1])
    return results

RCPARAMS = {
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

def compute_analysis_data(
    disease_name: str,
    df,
    interactome_graph,
    filtered_disease_dict: dict,
    drug_to_targets: dict,
    dist_matrix: np.ndarray,
    nodes: list,
    node_to_idx: dict,
    initial_drug_ids: List[str],
    num_qubits: int,
    num_samples: int = 10000,
) -> Tuple[str, Dict[str, float], np.ndarray, List[str]]:
    """
    Wrapper around analyze_disease_with_padding that returns
    (disease_name, z_dict, s_matrix, drug_ids).
    """
    return analyze_disease_with_padding(
        disease_name=disease_name,
        df=df,
        interactome_graph=interactome_graph,
        filtered_disease_dict=filtered_disease_dict,
        drug_to_targets=drug_to_targets,
        dist_matrix=dist_matrix,
        nodes=nodes,
        node_to_idx=node_to_idx,
        initial_drug_ids=initial_drug_ids,
        num_qubits=num_qubits,
        num_samples=num_samples,
    )

def plot_multi_disease_distributions(
    disease_data: Dict[str, Tuple[Dict[str, float], np.ndarray, List[str]]],
    figsize: Tuple = (16, 7.0),
    bins: int = 25,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    z-score and s histograms
    """
    plt.rcParams.update(RCPARAMS)

    diseases = list(disease_data.keys())
    n_diseases = len(diseases)

    fig, axes = plt.subplots(
        2,
        n_diseases,
        figsize=figsize,
        constrained_layout=True,
        sharex=False,
    )

    if n_diseases == 1:
        axes = axes.reshape(2, 1)

    for col, dname in enumerate(diseases):
        z_dict, s_matrix, drug_ids = disease_data[dname]
        n = len(drug_ids)

        z_vals = np.array([z_dict[did] for did in drug_ids])
        rows_idx, cols_idx = np.triu_indices(n, k=1)
        s_vals = s_matrix[rows_idx, cols_idx]

        ax_z = axes[0, col]
        ax_z.hist(
            z_vals,
            bins=bins,
            color="#4C72B0",
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )
        ax_z.axvline(x=0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_z.set_title(dname, fontsize=14, pad=8)
        ax_z.set_xlabel(r"$z(Y, X_i)$", fontsize=13, labelpad=4)
        if col == 0:
            ax_z.set_ylabel("Count", fontsize=13)
        ax_z.tick_params(axis="both", labelsize=11)
        ax_z.spines["top"].set_visible(False)
        ax_z.spines["right"].set_visible(False)

        ax_s = axes[1, col]
        ax_s.hist(
            s_vals,
            bins=bins,
            color="#DD8452",
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )
        ax_s.axvline(x=0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_s.set_xlabel(r"$s(X_i, X_j)$", fontsize=13, labelpad=4)
        if col == 0:
            ax_s.set_ylabel("Count", fontsize=13)
        ax_s.tick_params(axis="both", labelsize=11)
        ax_s.spines["top"].set_visible(False)
        ax_s.spines["right"].set_visible(False)

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")

    return fig
