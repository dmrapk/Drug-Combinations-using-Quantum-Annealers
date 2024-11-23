import numpy as np
import networkx as nx
import random 

def convert_to_graph(interactome_data):
    G = nx.Graph()
    for edge in interactome_data:
        node1, node2, _ = edge  
        if node1 != node2: 
            G.add_edge(int(node1), int(node2))
    return G

def filter_diseases(disease_gene_dict, interactome_graph):
    filtered_diseases = {}
    
    for disease_id, gene_ids in disease_gene_dict.items():
        gene_set = set(gene_ids)
        valid_nodes = set(gene_set)

        disconnected_nodes=[]
        for node in gene_set:
            paths_exist = True
            
            for other_node in gene_set:
                
                if node != other_node:
                    if not nx.has_path(interactome_graph, node, other_node):
                        paths_exist = False
                        disconnected_nodes.append(node)
                        break
        
        if len(disconnected_nodes)==1:
            valid_nodes.discard(disconnected_nodes[0])
            filtered_diseases[disease_id] = list(valid_nodes)
        
        if len(disconnected_nodes)==0:
            filtered_diseases[disease_id]=list(valid_nodes)

    return filtered_diseases

def generate_subgraph(interactome_graph, disease_gene_dict, n):

    random_disease_id = random.choice(list(disease_gene_dict.keys()))

    initial_nodes = set(disease_gene_dict[random_disease_id])

    subgraph_nodes = set(initial_nodes) 
    subgraph=interactome_graph.subgraph(subgraph_nodes)
    
    if (nx.is_connected(subgraph))==False:
        
        for node1 in initial_nodes:
            
            for node2 in initial_nodes:
                
                if node1 != node2 and not nx.has_path(subgraph, node1, node2):
                    
                    path = nx.shortest_path(interactome_graph, source=node1, target=node2)
                    subgraph_nodes.update(path)
                    
                    if len(subgraph_nodes)>n:
                        
                        return generate_subgraph(interactome_graph, disease_gene_dict, n)

    while len(subgraph_nodes) < n:
        
        neighbors = set()
        
        for node in subgraph_nodes:
            
            neighbors.update(set(interactome_graph.neighbors(node)))
        
        neighbors.difference_update(subgraph_nodes)
        
        if not neighbors:
            break
        
        new_node = random.choice(list(neighbors))
        subgraph_nodes.add(new_node)
    
    return subgraph_nodes,random_disease_id

def get_valid_drugs(drug_node_dict, graph):
    valid_drugs = []
    
    for drug_id, affected_nodes in drug_node_dict.items():
        if all(node in graph.nodes for node in affected_nodes):
            valid_drugs.append(drug_id)
        
    return valid_drugs 

def distance_matrix(graph):
    '''Calculates the distance matrix of a graph using the Smith–Waterman algorithm'''
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    
    nodes = list(graph.nodes)
    
    n = len(nodes)
    distance_mat = np.full((n, n), np.inf)  
    
    for i in range(n):
        for j in range(n):
            if i == j:
                distance_mat[i, j] = 0  
            else:
                node_i = nodes[i]
                node_j = nodes[j]
                distance_mat[i, j] = lengths[node_i].get(node_j, np.inf)
    
    return distance_mat, nodes

def distance_drug_d1(X, Y, distance_mat):

    sub_distances = distance_mat[np.ix_(X, Y)]
    
    min_distances = np.min(sub_distances, axis=0)
    
    summed_min_dists = np.sum(min_distances)
    
    return len(Y) / (summed_min_dists + 1)

def interaction_term(X, W, distance_matrix):

    sub_distances_xw = distance_matrix[np.ix_(X, W)] 
    sub_distances_wx = distance_matrix[np.ix_(W, X)]  
    
    min_distances_xw = np.min(sub_distances_xw, axis=0)  
    min_distances_wx = np.min(sub_distances_wx, axis=0) 
    
    summed_min_distances = np.sum(min_distances_xw) + np.sum(min_distances_wx)
    
    return (len(W) + len(X)) / summed_min_distances

def linear_coef(Xs,Y,beta, distance_matrix):
    
    linear_c=np.zeros(len(Xs))+beta

    for i,X in enumerate(Xs):
        linear_c[i]-=distance_drug_d1(X,Y, distance_matrix)
        
    return linear_c

def quadratic_coef(Xs, distance_matrix):
    quadratic_c=np.zeros((len(Xs),len(Xs)))
    
    for i,X1 in enumerate(Xs):
        for j in range(i+1,len(Xs)):
            X2=Xs[j]
            quadratic_c[i,j]=interaction_term(X1,X2, distance_matrix)
            
    return quadratic_c

def generate_disease_dict(diseases,interactome_graph):
    disease_gene_dict = {}

    for index, row in diseases.iterrows():
        disease_id = row['# Disease ID'] 
        gene_id = row['Gene ID']
    
        if gene_id in interactome_graph:
            if disease_id not in disease_gene_dict:
                disease_gene_dict[disease_id] = []
        
            disease_gene_dict[disease_id].append(gene_id)
            
    return disease_gene_dict

def generate_targets(drug_target_data):
    
    drug_to_targets = {}

    target_sets = []

    for drug, target in drug_target_data:
        current_targets = drug_to_targets.get(drug, []) + [target]
    
        if not any(set(current_targets) == target_set for target_set in target_sets):
            if drug not in drug_to_targets:
                drug_to_targets[drug] = []

            drug_to_targets[drug].append(target)

            target_sets.append(set(current_targets))
            
    return drug_to_targets, target_sets

def generate_problem(graph_size, drug_to_targets, num_spins, gamma, interactome_graph, filtered_disease_dict):
                     
    graph_nodes,disease_id = generate_subgraph(interactome_graph, filtered_disease_dict, graph_size)

    graph = interactome_graph.subgraph(graph_nodes)
    valid_drugs = get_valid_drugs(drug_to_targets, graph)
    
    while len(valid_drugs)<num_spins:
        graph_nodes, disease_id = generate_subgraph(interactome_graph, filtered_disease_dict, graph_size)
        graph = interactome_graph.subgraph(graph_nodes)
        valid_drugs = get_valid_drugs(drug_to_targets, graph)
        
    chosen_drugs = np.random.choice(valid_drugs, num_spins, replace=False)

    Y=np.array([node for node in filtered_disease_dict[disease_id]])
    Xs = np.empty(len(chosen_drugs), dtype=object) 
    for i, drug in enumerate(chosen_drugs):
        Xs[i] = np.array([node for node in drug_to_targets[drug]]) 
    
    distance_mat, node_list = distance_matrix(graph)

    return graph, Y , Xs, distance_mat, node_list
