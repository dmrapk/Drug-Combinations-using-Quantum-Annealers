import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from typing import Dict, List, Tuple

def convert_to_graph(interactome_data) -> nx.Graph:
    """
    Build an undirected NetworkX graph from an edge list.
    Each record in interactome_data is expected to contain (node1, node2, ...).
    Self-edges are ignored. Node IDs are cast to int.
    """
    G = nx.Graph()
    for edge in interactome_data:
        node1, node2, _ = edge  
        if node1 != node2: 
            G.add_edge(int(node1), int(node2))
    return G

def filter_diseases(disease_gene_dict: Dict, interactome_graph: nx.Graph) -> Dict:
    """
    Ignores diseases with more than one disconnected gene to avoid interactions between nodes in different network clusters.
    """
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

def generate_disease_dict(diseases: pd.DataFrame, interactome_graph: nx.Graph) -> Dict:
    """
    Create a mapping {disease_id: [gene_id, ...]} from the diseases DataFrame.
    Only genes that are present as nodes in interactome_graph are included.
    """
    disease_gene_dict = {}

    for index, row in diseases.iterrows():
        disease_id = row['# Disease ID'] 
        gene_id = row['Gene ID']
    
        if gene_id in interactome_graph:
            if disease_id not in disease_gene_dict:
                disease_gene_dict[disease_id] = []
        
            disease_gene_dict[disease_id].append(gene_id)
            
    return disease_gene_dict

def load_and_merge_combinations(combos_path: str, diseases: pd.DataFrame) -> pd.DataFrame:
    """
    Load drug combination TSV and merge human readable disease names from the `diseases`
    """
    df = pd.read_csv(combos_path, sep='\t')

    diseases = diseases.copy()
    diseases.columns = diseases.columns.str.strip()
    diseases = diseases.rename(columns={'# Disease ID': 'Disease ID'})

    df['Drug Combination'] = df['Drug Combination'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    df['Num Drugs'] = df['Drug Combination'].apply(len)

    disease_names = diseases[['Disease ID', 'Disease Name']].drop_duplicates()
    df = df.merge(disease_names, on='Disease ID', how='left')

    return df

def get_drug_combinations_and_unique_drugs(disease_name: str, df: pd.DataFrame):
    """
    For a given disease name, return a list of drug combinations and a sorted list of unique drugs.
    """
    subset = df[df['Disease Name'] == disease_name]
    
    combinations = subset['Drug Combination'].tolist()
    
    unique_drugs = sorted({
        drug 
        for combo in combinations 
        for drug in combo
    })
    
    return combinations, unique_drugs

def get_ground_truth_combinations(disease_name: str, df: pd.DataFrame) -> set[frozenset]:
    """
    For a given disease name, return a set of frozensets representing the ground truth drug combinations.
    """
    disease_data = df[df['Disease Name'] == disease_name]
    return {frozenset(combo) for combo in disease_data['Drug Combination']}

def plot_combination_info(df, top_n=10):
    """
    Generate plots summarizing drug combination data.
    """
    combo_counts = df['Disease Name'].value_counts().reset_index()
    combo_counts.columns = ['Disease Name', 'Count']

    grouped = df.groupby(['Disease Name', 'Num Drugs']).size().reset_index(name='Count')

    pivot_all = grouped.pivot(index='Disease Name', columns='Num Drugs', values='Count').fillna(0)
    pivot_all = pivot_all.loc[pivot_all.sum(axis=1).sort_values(ascending=False).index]

    plt.figure(figsize=(14, 6))
    bottom = None
    colors = sns.color_palette("viridis", len(pivot_all.columns))
    for idx, num_drugs in enumerate(pivot_all.columns):
        plt.bar(pivot_all.index, pivot_all[num_drugs], label=f'{num_drugs} drugs', bottom=bottom, color=colors[idx])
        bottom = pivot_all[num_drugs] if bottom is None else bottom + pivot_all[num_drugs]
    plt.xticks(rotation=90)
    plt.xlabel('Disease Name')
    plt.ylabel('Number of Combinations')
    plt.title('Drug Combinations per Disease (All)')
    plt.legend(title='Number of Drugs')
    plt.tight_layout()
    #plt.savefig('DrugCombsperDisease.png')
    plt.show()

    top10 = combo_counts.head(top_n)
    top10_names = top10['Disease Name'].tolist()
    filtered_grouped = grouped[grouped['Disease Name'].isin(top10_names)]

    pivot_top10 = filtered_grouped.pivot(index='Disease Name', columns='Num Drugs', values='Count').fillna(0)
    pivot_top10 = pivot_top10.reindex(top10['Disease Name'])

    plt.figure(figsize=(12, 6))
    bottom = None
    colors = sns.color_palette("magma", len(pivot_top10.columns))
    for idx, num_drugs in enumerate(pivot_top10.columns):
        plt.bar(pivot_top10.index, pivot_top10[num_drugs], label=f'{num_drugs} drugs', bottom=bottom, color=colors[idx])
        bottom = pivot_top10[num_drugs] if bottom is None else bottom + pivot_top10[num_drugs]
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Disease Name')
    plt.ylabel('Number of Combinations')
    plt.title(f'Top {top_n} - Drug Combinations by Number of Drugs')
    plt.legend(title='Number of Drugs')
    plt.tight_layout()
    #plt.savefig('top10.png')
    plt.show()

    disease_to_drugs = df.groupby('Disease Name')['Drug Combination'].apply(lambda combos: set(drug for combo in combos for drug in combo))
    drug_counts = disease_to_drugs.apply(len).reset_index()
    drug_counts.columns = ['Disease Name', 'Unique Drug Count']
    drug_counts = drug_counts.sort_values(by='Unique Drug Count', ascending=False)

    plt.figure(figsize=(14, 6))
    sns.barplot(data=drug_counts, x='Disease Name', y='Unique Drug Count', color='mediumseagreen')
    plt.xticks(rotation=90)
    plt.xlabel('Disease Name')
    plt.ylabel('Number of Unique Drugs')
    plt.title('Number of Unique Drugs per Disease')
    plt.tight_layout()
    #plt.savefig('uniqueDrugs.png')
    plt.show()
