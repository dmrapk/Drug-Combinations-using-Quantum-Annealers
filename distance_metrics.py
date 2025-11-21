import numpy as np
import networkx as nx
from collections import defaultdict
import random as random
from typing import List, Tuple, Dict, Tuple, Any
import pandas as pd 
from pathlib import Path

def precompute_apsp(G: nx.Graph) -> Tuple[np.ndarray, List, Dict]:
    """
    Precompute all-pairs shortest-path distances for graph G.
    """
    nodes = list(G.nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)
    
    dist_matrix = np.full((n_nodes, n_nodes), np.inf)
    np.fill_diagonal(dist_matrix, 0)
    
    for i, node in enumerate(nodes):
        try:
            lengths = nx.shortest_path_length(G, node)
            for target, dist in lengths.items():
                j = node_to_idx[target]
                dist_matrix[i, j] = dist
        except nx.NetworkXError:
            continue
    
    return dist_matrix, nodes, node_to_idx

def load_apsp(path: str | Path) -> Tuple[np.ndarray, List[Any], Dict[Any, int]]:
    """
    Load APSP data.

    Returns:
        dist_matrix (np.ndarray), nodes (list), node_to_idx (dict)
    """
    path = Path(path)
    fmt = path.suffix.lstrip('.').lower()
    if fmt == 'npz':
        with np.load(path, allow_pickle=True) as data:
            dist_matrix = data['dist_matrix']
            nodes_arr = data['nodes'] 
            node_to_idx_arr = data['node_to_idx']  
            try:
                nodes = list(nodes_arr.tolist())
            except Exception:
                nodes = list(nodes_arr.item())
            try:
                node_to_idx = node_to_idx_arr.item()
                if not isinstance(node_to_idx, dict):
                    node_to_idx = dict(node_to_idx)
            except Exception:
                node_to_idx = dict(node_to_idx_arr.tolist())
        return dist_matrix, nodes, node_to_idx
    else:
        raise ValueError("Unsupported file type. Expected .npz")
    
def calculate_zscores_vectorized(
    G,
    Xs: List[List],
    Y: List,
    dist_matrix: np.ndarray,
    nodes: List,
    node_to_idx: Dict,
    num_samples: int = 2000,
    cap: int = 4
) -> Dict[int, float]:
    """
    Compute z-scores for each drug X in Xs relative to Y using degree-preserving
    random sampling. Returns a dict mapping each X index to its z-score.
    """
    n_nodes = len(nodes)

    Y_indices = np.array([node_to_idx[y] for y in Y if y in node_to_idx])
    Xs_indices = [np.array([node_to_idx[x] for x in X if x in node_to_idx]) for X in Xs]

    degree_to_indices = defaultdict(list)
    for idx, node in enumerate(nodes):
        degree_to_indices[G.degree(node)].append(idx)
    
    Y_degrees = [G.degree(y) for y in Y]
    Y_samples = np.zeros((len(Y), num_samples), dtype=int)
    for i, d in enumerate(Y_degrees):
        possible = degree_to_indices[d]
        Y_samples[i, :] = np.random.choice(possible, size=num_samples, replace=True)
    
    z_scores = {}
    for x_idx, X_indices in enumerate(Xs_indices):
        X_degrees = [G.degree(nodes[i]) for i in X_indices]
        X_samples = np.zeros((len(X_indices), num_samples), dtype=int)
        for j, d in enumerate(X_degrees):
            possible = degree_to_indices[d]
            X_samples[j, :] = np.random.choice(possible, size=num_samples, replace=True)
        
        d_observed = compute_mean_distance(Y_indices, X_indices, dist_matrix,cap)
        
        ref_dists = []
        for s in range(num_samples):
            rand_Y = Y_samples[:, s]
            rand_X = X_samples[:, s]
            ref_dists.append(compute_mean_distance(rand_Y, rand_X, dist_matrix,cap))
        
        mu = np.mean(ref_dists)
        sigma = np.std(ref_dists)
        z_scores[x_idx] = (d_observed - mu)/sigma if sigma != 0 else np.inf
    
    return z_scores

def compute_mean_distance(src_indices: np.ndarray, tgt_indices: np.ndarray, dist_matrix: np.ndarray, cap: int) -> float:
    """
    Mean of pairwise shortest-path distances between src_indices and tgt_indices.
    """
    if src_indices is None or tgt_indices is None:
        return cap

    src_indices = np.asarray(src_indices, dtype=int)
    tgt_indices = np.asarray(tgt_indices, dtype=int)

    if src_indices.size == 0 or tgt_indices.size == 0:
        return cap

    block = dist_matrix[np.ix_(src_indices, tgt_indices)]
    flat = block.ravel()
    finite_mask = np.isfinite(flat)
    if np.any(finite_mask):
        return float(np.mean(flat[finite_mask]))

    return cap


def compute_mean_intra_distance(indices: np.ndarray, dist_matrix: np.ndarray,cap: int) -> float:
    """
    Mean of pairwise shortest-path distances among unordered distinct pairs in `indices`.
    """
    if indices is None:
        return 0.0
    indices = np.asarray(indices, dtype=int)
    if indices.size < 2:
        return 0.0

    rows, cols = np.triu_indices(len(indices), k=1)
    if rows.size == 0:
        return 0.0

    pair_dists = dist_matrix[np.ix_(indices[rows], indices[cols])]
    flat = pair_dists.ravel()
    finite_mask = np.isfinite(flat)
    if np.any(finite_mask):
        return float(np.mean(flat[finite_mask]))

    return cap


def calculate_s_matrix(
    Xs: List[List],
    dist_matrix: np.ndarray,
    node_to_idx: Dict[Any, int],
    cap: int=4
) -> np.ndarray:
    """
    Compute topological separation measure matrix s_{ij} for list of target-sets Xs:
      s(X_i, X_j) = <d_{X_i X_j}> - 0.5*(<d_{X_i X_i}> + <d_{X_j X_j}>)
    """
    Xs_indices = [np.array([node_to_idx[x] for x in X if x in node_to_idx], dtype=int) for X in Xs]
    n = len(Xs_indices)

    mean_intra = np.zeros(n, dtype=float)
    for i in range(n):
        mean_intra[i] = compute_mean_intra_distance(Xs_indices[i], dist_matrix, cap)

    s_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            mean_Xi_Xj = compute_mean_distance(Xs_indices[i], Xs_indices[j], dist_matrix, cap)
            s_val = float(mean_Xi_Xj - 0.5 * (mean_intra[i] + mean_intra[j]))
            s_matrix[i, j] = s_val
            s_matrix[j, i] = s_val

    for i in range(n):
        if len(Xs_indices[i]) >= 2:
            s_matrix[i, i] = 0.0
        else:
            s_matrix[i, i] = 0.0

    return s_matrix


def analyze_disease_with_padding(
    disease_name: str,
    df: pd.DataFrame,
    interactome_graph,
    filtered_disease_dict: dict,
    drug_to_targets: dict,
    dist_matrix: np.ndarray,
    nodes: list,
    node_to_idx: dict,
    initial_drug_ids: List[str],
    num_qubits: int,
    num_samples: int = 10000
) -> Tuple[List[str], Dict[str, float], np.ndarray, List[str]]:
    """
    Pads `initial_drug_ids` to length `num_qubits` with other random drugs,
    computes Z-scores and the s-matrix for the padded drug list.
    """

    base_drugs = [did for did in initial_drug_ids if did in drug_to_targets]

    if len(base_drugs) > num_qubits:
        base_drugs = base_drugs[:num_qubits]

    elif len(base_drugs) < num_qubits:
        candidates = [did for did in drug_to_targets if did not in base_drugs]
        needed = num_qubits - len(base_drugs)
        if len(candidates) < needed:
            raise ValueError(f"Not enough extra drugs: need {needed}, have {len(candidates)}")
        base_drugs += random.sample(candidates, needed)

    drug_ids = base_drugs

    Xs = [[x for x in drug_to_targets[did] if x in node_to_idx] for did in drug_ids]

    matches = df[df['Disease Name'] == disease_name]['Disease ID'].unique().tolist()
    if not matches:
        raise ValueError(f"Disease '{disease_name}' not found in DF")
    disease_id = matches[0]
    Y = [y for y in filtered_disease_dict.get(disease_id, []) if y in node_to_idx]

    cap = 2.0 * float(np.max(dist_matrix[np.isfinite(dist_matrix)]))

    raw_z = calculate_zscores_vectorized(
        G=interactome_graph,
        Xs=Xs,
        Y=Y,
        dist_matrix=dist_matrix,
        nodes=nodes,
        node_to_idx=node_to_idx,
        num_samples=num_samples,
        cap=cap
    )
    if isinstance(raw_z, dict):
        z_scores = np.array([raw_z[i] for i in range(len(drug_ids))])
    else:
        z_scores = np.array(raw_z)
    z_dict = {did: float(z_scores[i]) for i, did in enumerate(drug_ids)}
    

    s_matrix = calculate_s_matrix(Xs=Xs, dist_matrix=dist_matrix, node_to_idx=node_to_idx, cap= cap)
    return disease_name, z_dict, s_matrix, drug_ids

