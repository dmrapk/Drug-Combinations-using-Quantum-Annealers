import numpy as np
import networkx as nx
from collections import defaultdict

def precompute_apsp(G):
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

def calculate_zscores_vectorized(G, Xs, Y, dist_matrix, nodes, node_to_idx, num_samples=300):
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

def calculate_s_matrix(Xs, Y, dist_matrix, nodes, node_to_idx):
    
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

def compute_mean_distance(src_indices, tgt_indices, dist_matrix):
    mask = np.ix_(src_indices, tgt_indices)
    distances = dist_matrix[mask]
    valid = np.isfinite(distances)
    return distances[valid].mean() if valid.any() else 0.0

def compute_mean_intra_distance(indices, dist_matrix):
    if len(indices) < 2:
        return 0.0
    rows, cols = np.triu_indices(len(indices), k=1)
    distances = dist_matrix[indices[rows], indices[cols]]
    valid = np.isfinite(distances)
    return distances[valid].mean() if valid.any() else 0.0


def generate_hard_problem(interactome_graph, filtered_disease_dict, drug_to_targets,
                            dist_matrix, nodes, node_to_idx, num_samples=300):
    disease_ids = list(filtered_disease_dict.keys())
    disease_id = np.random.choice(disease_ids)
    Y = [y for y in filtered_disease_dict[disease_id] if y in node_to_idx]
    
    if not Y: 
        return disease_id, {}, np.array([]), []
    
    drug_ids = list(drug_to_targets.keys())
    Xs = [[x for x in drug_to_targets[did] if x in node_to_idx] for did in drug_ids]
    
    valid_drug_mask = [len(x) > 0 for x in Xs]
    valid_drug_ids = [did for did, valid in zip(drug_ids, valid_drug_mask) if valid]
    Xs = [x for x in Xs if len(x) > 0]
    
    if not Xs:  
        return disease_id, {}, np.array([]), []
    
    try:
        z_scores = calculate_zscores_vectorized(
            G=interactome_graph,
            Xs=Xs,
            Y=Y,
            dist_matrix=dist_matrix,
            nodes=nodes,
            node_to_idx=node_to_idx,
            num_samples=num_samples
        )
    except KeyError as Erro:
        raise RuntimeError(f"Missing node in graph: {Erro}") from Erro
    
    s_matrix = calculate_s_matrix(
        Xs=Xs,
        Y=Y,
        dist_matrix=dist_matrix,
        nodes=nodes,
        node_to_idx=node_to_idx
    )
    
    negative_indices = [i for i in z_scores if z_scores[i] < 0]
    filtered_z = {valid_drug_ids[i]: z_scores[i] for i in negative_indices}
    filtered_drug_ids = [valid_drug_ids[i] for i in negative_indices]
    
    if filtered_drug_ids:
        filtered_s = s_matrix[np.ix_(negative_indices, negative_indices)]
    else:
        filtered_s = np.array([])
    
    return disease_id, filtered_z, filtered_s, filtered_drug_ids

#-----------------------//----------------------#
# Example Usage

dist_matrix, nodes, node_to_idx = precompute_apsp(interactome_graph)

disease_id, z_dict, s_mat, drug_ids = generate_hard_problem(
    interactome_graph,
    filtered_disease_dict,
    drug_to_targets,
    dist_matrix,
    nodes,
    node_to_idx
)

print(f"Analyzed disease: {disease_id}")
print(f"Negative z-scores found: {len(z_dict)}")