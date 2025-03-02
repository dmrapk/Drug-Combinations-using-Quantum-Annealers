from problem_generation import convert_to_graph, generate_disease_dict, generate_targets, filter_diseases, generate_problem
from Scoring import calculate_zscores, bf_get_best_therapy, score_therapy
from DWave_Hardware import get_qubo_from_problem, get_DWaveSampler_Best_Therapy
import numpy as np
#Load data here

#diseases = 
#drug_target_data = 
#interactome_data = 

interactome_graph = convert_to_graph(interactome_data)
    
disease_gene_dict = generate_disease_dict(diseases,interactome_graph)
    
drug_to_targets, target_sets = generate_targets(drug_target_data)
    
filtered_disease_dict = filter_diseases(disease_gene_dict, interactome_graph)
    
#Parameters

#beta=0
#gamma=0.75
#graph_size=300
#num_spins=10

graph, Y , Xs, distance_mat, node_list = generate_problem(graph_size, drug_to_targets, num_spins, gamma, interactome_graph, filtered_disease_dict)

z_dict=calculate_zscores(graph,Xs,Y)

score, best_therapy, worst_score = bf_get_best_therapy(graph, Xs, Y, z_dict)

Q = get_qubo_from_problem(Xs,Y,distance_mat, gamma, beta, node_list)

targets, target_nodes = get_DWaveSampler_Best_Therapy(Q, Xs)

scoreqa, s, z = score_therapy(graph, target_nodes, targets, Y, z_dict)

score_ratio=(scoreqa+np.abs(worst_score))/(score+np.abs(worst_score))
