import numpy as np
import networkx as nx
from collections import defaultdict
import random as random
from typing import List, Tuple, Dict
import pandas as pd


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

def calculate_zscores_vectorized(
    G,
    Xs: List[List],
    Y: List,
    dist_matrix: np.ndarray,
    nodes: List,
    node_to_idx: Dict,
    num_samples: int = 2000
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
        
        d_observed = compute_mean_distance(Y_indices, X_indices, dist_matrix)
        
        ref_dists = []
        for s in range(num_samples):
            rand_Y = Y_samples[:, s]
            rand_X = X_samples[:, s]
            ref_dists.append(compute_mean_distance(rand_Y, rand_X, dist_matrix))
        
        mu = np.mean(ref_dists)
        sigma = np.std(ref_dists)
        z_scores[x_idx] = (d_observed - mu)/sigma if sigma != 0 else np.inf
    
    return z_scores

def calculate_s_matrix(
    Xs: List[List],
    Y: List,
    dist_matrix: np.ndarray,
    nodes: List,
    node_to_idx: Dict
) -> np.ndarray:
    """
    Compute the topological separation measure s_{ij} for a drug set Xs relative to Y using its mean shortest distances.
    Returns an (n x n) symmetric matrix with zeros on the diagonal.
    """    
    Y_indices = np.array([node_to_idx[y] for y in Y if y in node_to_idx])
    Xs_indices = [np.array([node_to_idx[x] for x in X if x in node_to_idx]) for X in Xs]

    n = len(Xs_indices)
    
    mean_Y_Y = compute_mean_intra_distance(Y_indices, dist_matrix)
    mean_Y_X = np.array([compute_mean_distance(Y_indices, Xi, dist_matrix) for Xi in Xs_indices])
    
    s_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                s_matrix[i, j] = 0.0  
            else:
                mean_Xi_Xj = compute_mean_distance(Xs_indices[i], Xs_indices[j], dist_matrix)
                s = mean_Y_X[i] - 0.5*(mean_Y_Y + mean_Xi_Xj)
                s_matrix[i, j] = s
                s_matrix[j, i] = s
    
    return s_matrix

def compute_mean_distance(src_indices, tgt_indices, dist_matrix: np.ndarray) -> float:
    """
    Compute mean of pairwise distances between source indices and target indices using the precomputed distances matrix.
    Returns 0.0 if there are no finite distances.
    """
    mask = np.ix_(src_indices, tgt_indices)
    distances = dist_matrix[mask]
    valid = np.isfinite(distances)
    return distances[valid].mean() if valid.any() else 0.0

def compute_mean_intra_distance(indices, dist_matrix: np.ndarray) -> float:
    """
    Compute mean pairwise distance among indices. Returns 0.0 for <2 indices.
    """
    if len(indices) < 2:
        return 0.0
    rows, cols = np.triu_indices(len(indices), k=1)
    distances = dist_matrix[indices[rows], indices[cols]]
    valid = np.isfinite(distances)
    return distances[valid].mean() if valid.any() else 0.0

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
    num_samples: int = 2000
) -> Tuple[List[str], Dict[str, float], np.ndarray, List[str]]:
    """
    Pads `initial_drug_ids` to length `num_qubits` with other random drugs,
    then computes Z-scores and s-matrix for exactly `num_qubits` drugs.
    """

    base_drugs = [did for did in initial_drug_ids if did in drug_to_targets]
    if len(base_drugs) > num_qubits:
        base_drugs = base_drugs[:num_qubits]
    elif len(base_drugs) < num_qubits:
        candidates = [did for did in drug_to_targets if did not in base_drugs]
        needed = num_qubits - len(base_drugs)
        if len(candidates) < needed:
            raise ValueError(
                f"Not enough extra drugs: need {needed}, have {len(candidates)}"
            )
        base_drugs += random.sample(candidates, needed)

    drug_ids = base_drugs  

    Xs = [[x for x in drug_to_targets[did] if x in node_to_idx] for did in drug_ids]

    matches = df[df['Disease Name'] == disease_name]['Disease ID'].unique().tolist()
    if not matches:
        raise ValueError(f"Disease '{disease_name}' not found in DF")
    disease_id = matches[0]
    Y = [y for y in filtered_disease_dict.get(disease_id, []) if y in node_to_idx]

    raw_z = calculate_zscores_vectorized(
        G=interactome_graph,
        Xs=Xs,
        Y=Y,
        dist_matrix=dist_matrix,
        nodes=nodes,
        node_to_idx=node_to_idx,
        num_samples=num_samples
    )

    if isinstance(raw_z, dict):
        z_scores = np.array([raw_z[i] for i in range(len(drug_ids))])
    else:
        z_scores = np.array(raw_z)
    z_dict = {did: float(z_scores[i]) for i, did in enumerate(drug_ids)}

    s_matrix = calculate_s_matrix(
        Xs=Xs,
        Y=Y,
        dist_matrix=dist_matrix,
        nodes=nodes,
        node_to_idx=node_to_idx
    )

    return disease_name, z_dict, s_matrix, drug_ids