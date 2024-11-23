import numpy as np
import networkx as nx
from dwave.system import DWaveSampler, EmbeddingComposite
import ast
from problem_generation import linear_coef, quadratic_coef

def get_qubo_from_problem(Xs,Y,distance_mat,gamma,beta,node_list):
    
    Y2=np.array([node_list.index(node) for node in Y])
        
    Xs2 = np.empty(len(Xs), dtype=object) 
    for i, drug in enumerate(Xs):
        Xs2[i] = np.array([node_list.index(node) for node in drug]) 
    linear_coefficients = linear_coef(Xs2,Y2,beta, distance_mat) 
    quadratic_coefficients = gamma * quadratic_coef(Xs2, distance_mat)

    num_vars = len(linear_coefficients)

    Q = {}
    for i in range(num_vars):
        Q[(i, i)] = linear_coefficients[i]
    
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            Q[(i, j)] = quadratic_coefficients[i, j]
            
    return Q
    
def get_DWaveSampler_Best_Therapy(Q, Xs,num_reads=100):

    sampler = EmbeddingComposite(DWaveSampler())
    sampleset = sampler.sample_qubo(Q, num_reads)
    samples = sampleset.samples()
    energies = sampleset.record.energy

    dict_object = ast.literal_eval(str(samples[0]))
    values = np.array(list(dict_object.values()))
    indices = np.where(values == 1)[0]
    target_nodes=[]
    
    for j in indices:
        target_nodes.append(Xs[j])
    
    targets=[]
    for i, x in enumerate(Xs):
        if any(np.array_equal(x, target) for target in target_nodes):
            targets.append(i)
            
    return targets,target_nodes
    