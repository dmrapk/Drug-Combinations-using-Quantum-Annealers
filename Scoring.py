import numpy as np
import networkx as nx
import itertools

def compute_d_Y_X(G, Y, X):

    distances = []
    for y in Y:
        shortest_to_X = [nx.shortest_path_length(G, source=y, target=x) for x in X if nx.has_path(G, y, x)]
        if shortest_to_X:  
            distances.append(min(shortest_to_X))
    
    if distances:
        return np.mean(distances)
    else:
        return float('inf')  

def generate_reference_distribution(G, Y, X, num_samples=1000):

    degrees_Y = [G.degree(y) for y in Y]
    degrees_X = [G.degree(x) for x in X]
    
    reference_distances = []
    for _ in range(num_samples):
        random_Y = [node for degree in degrees_Y for node in np.random.choice([n for n in G.nodes if G.degree(n) == degree], 1)]
        random_X = [node for degree in degrees_X for node in np.random.choice([n for n in G.nodes if G.degree(n) == degree], 1)]
        d_random = compute_d_Y_X(G, random_Y, random_X)
        reference_distances.append(d_random)
    
    return np.mean(reference_distances), np.std(reference_distances)

def z_score(G, Y, X, num_samples=200):

    d_Y_X = compute_d_Y_X(G, Y, X)
    mu, sigma = generate_reference_distribution(G, Y, X, num_samples)
    
    if sigma == 0:
        return float('inf')  
        
    return (d_Y_X - mu) / sigma

def mean_shortest_distance(G, Y, X, pair_type="Y-Y"):
    distances = []
    
    if pair_type == "Y-X":
        for y in Y:
            for x in X:
                if nx.has_path(G, y, x):
                    distances.append(nx.shortest_path_length(G, source=y, target=x))
    
    elif pair_type == "Y-Y":
        for y1, y2 in itertools.combinations(Y, 2):
            if nx.has_path(G, y1, y2):
                distances.append(nx.shortest_path_length(G, source=y1, target=y2))
    
    elif pair_type == "X-X":
        for x1, x2 in itertools.combinations(X, 2):
            if nx.has_path(G, x1, x2):
                distances.append(nx.shortest_path_length(G, source=x1, target=x2))
    
    if distances:
        return np.mean(distances)
    else:
        return 0  

def calculate_mean_s_measure(G, Y, Xs):
    n = len(Xs)
    if n == 1 or n==0:
        return 0
    total_s = 0
    number_of_pairs=0
    mean_distance_Y_Y = mean_shortest_distance(G, Y, Y, pair_type="Y-Y")

    for i in range(n):
        mean_distance_Y_X = mean_shortest_distance(G, Y, Xs[i], pair_type="Y-X")

        for j in range(i + 1, n):
            mean_distance_X_X = mean_shortest_distance(G, Xs[i], Xs[j], pair_type="X-X")

            s = mean_distance_Y_X-(mean_distance_Y_Y + mean_distance_X_X)/2
            total_s += s
            number_of_pairs+=1
            
    return total_s/number_of_pairs

def score_therapy(G, Xs, targets, Y, z_dict):
    Z=0
    for i in targets:
        Z+=z_dict[i]
        
    S = calculate_mean_s_measure(G, Y, Xs)
    score = S-(Z)/len(Xs)
    
    return score, S, Z/len(Xs)

def bf_get_best_therapy(graph, Xs, Y, z_dict):
    
    best_score = -np.inf
    worst_score = np.inf

    best_therapy = []
    for i in range(1, len(Xs) + 1): 
        combs = itertools.combinations(range(len(Xs)), i)
        for combination in combs:
            nodes_combination = [Xs[j] for j in combination]
            s = calculate_mean_s_measure(graph, Y, nodes_combination)
            
            z = np.mean([z_dict[j] for j in combination])
            
            score = s - (z)
            
            if score > best_score:
                best_score = score
                best_therapy = nodes_combination

            if score < worst_score:
                worst_score = score

    return best_score, best_therapy, worst_score

def calculate_zscores(graph,Xs,Y,num_samples=100):
    z_dict = {}
    for i, X in enumerate(Xs):
        zsc = z_score(graph, Y, X, num_samples)
        z_dict[i]  = zsc
    return z_dict
    