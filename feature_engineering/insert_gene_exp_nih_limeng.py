"""
This script does two things:
1. extract gene expression data from the graph, then add to NIH_part_SNPs_features_nodup.csv.
2. extract disease scores from NIH_SNPs_features.csv, then add to NIH_part_SNPs_features_nodup.csv.
"""

import argparse
import networkx as nx
import pandas as pd
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-data_dir',
                        default='../../data/original/NIH_part_SNPs_features_nodup.csv',
                        required=False,
                        help='directory of gene expression graph')

    parser.add_argument('-data_dir_with_disease_scores',
                        default='../../data/original/NIH_SNPs_features.csv',
                        required=False,
                        help='directory of gene expression graph')

    parser.add_argument('-graph_dir',
                        #default='../../data/original/string_ERpos.gexf',
                        default='../../data/original/string_ERpos_neighbours.gexf',
                        required=False,
                        help='directory of the dataset')

    parser.add_argument('-out_dir',
                        default='../../data/output/NIH_part_SNPs_features_nodup_new.csv',
                        required=False,
                        help='directory of output dataset')    
                 
    return parser.parse_args()

def get_gene_exp(graph_dict, ensembl_id):
    """
    Retrieve the gene expression from the graph according to the ensembl id,
    then return the corresponding gene expression as integer.  
    """
    return int(float(graph_dict[ensembl_id]))

def get_disease_scores(ensembl_id, ensembl_dict_reversed, diseases_score_dict, disgenet_score_dict, malacards_score_dict):
    index = ensembl_dict_reversed[ensembl_id]
    diseases_score = diseases_score_dict[index]
    disgenet_score = disgenet_score_dict[index]
    malacards_score = malacards_score_dict[index]
    return diseases_score, disgenet_score, malacards_score

if __name__=="__main__":
    args=get_args()
    graph_dir = args.graph_dir
    data_dir = args.data_dir
    out_dir = args.out_dir
    data_dir_with_disease_scores = args.data_dir_with_disease_scores
    print('------------------------------------------')
    print('graph directory:', graph_dir)
    print('data directory:', data_dir)
    print('output data directory:', out_dir)
    print('data directory to get disease scores:', data_dir_with_disease_scores)
    gene_exp_graph = nx.read_gexf(graph_dir) # read gene expressions as graph
    ensembl_ids = nx.get_node_attributes(gene_exp_graph, "ensembl") # get ensemble ids
    gene_exps = nx.get_node_attributes(gene_exp_graph, "gene_exp") # get gene expressions
    print('number of nodes in graph:',len(ensembl_ids))
    print('number of corresponding gene expressions:', len(gene_exps))
    assert(len(ensembl_ids)==len(gene_exps))
    
    # making sure that the two lists are matched pairs, so accessing the dictionaries with
    # the same key when generating the lists
    print('Processing graph data to dictionary...')
    ensembl_id_list = []
    gene_exp_list = []
    with tqdm(total=len(ensembl_ids)) as pbar:
        for node_id in ensembl_ids.keys():
            ensembl_id_list.append(ensembl_ids[node_id])
            gene_exp_list.append(gene_exps[node_id])
            pbar.update(1)
    graph_dict = dict(zip(ensembl_id_list, gene_exp_list)) # put them into dictionary
    print('size of combined dictionary:', len(graph_dict))

    # read the dataframe from csv file without gene expressions
    #df = pd.read_csv(data_dir, delim_whitespace=True)
    df = pd.read_csv(data_dir)
    column_names = ['ensembl',
                    'wildtype_value', 
                    'mutant_value', 
                    'confidence', 
                    'core', 
                    'nis', 
                    'interface', 
                    'hbo', 
                    'sbr', 
                    'aromatic', 
                    'hydrophobic',  
                    'blosum62_mu', 
                    'blosum62_wt', 
                    'helix', 
                    'coil', 
                    'sheet', 
                    'entropy', 
                    #'diseases_score', 
                    #'disgenet_score', 
                    #'malacards_score', 
                    #'gene_exp',
                    'p_value']
    df = df[column_names]
    print('number of samples in original dataset:', df.shape[0])    
    print('number of features in original dataset:',df.shape[1])

    #print('entries with -1 p-values:')
    #print(df[df.p_value == -1])

    df = df[df.p_value != -1] # exclude the rows which have missing p-values
    print('number of samples after dropping samples with missing p-values:', df.shape[0])

    # read disease scores and add to dataframe
    df_disease = pd.read_csv(data_dir_with_disease_scores, delim_whitespace=True)
    column_names_disease = ['ensembl', 'diseases_score', 'disgenet_score', 'malacards_score']
    df_disease = df_disease[column_names_disease]
    print('shape of disease score dataframe:', df_disease.shape)
    df_disease = df_disease.drop_duplicates() # drop duplicate rows since the same ensembl has the same disease score
    print('shape of disease score dataframe after dropping duplicates:', df_disease.shape)
    disease_dict = df_disease.to_dict()
    ensembl_dict = disease_dict['ensembl']
    diseases_score_dict = disease_dict['diseases_score']
    disgenet_score_dict = disease_dict['disgenet_score_score']
    malacards_score_dict = disease_dict['malacards_score_score']
    ensembl_dict_reversed = dict((v,k) for k,v in ensembl_dict.items())
    print(ensembl_dict_reversed)
    
    ensp_ids = list(df['ensembl']) # get ENSP-IDs in the dataset as iterable

    print('Processing the .csv file...')
    gene_exp_list = [] # create an empty list for gene expressions
    diseases_score_list = []
    disgenet_score_list = []
    malacards_score_list = []  
    with tqdm(total=df.shape[0]) as pbar:
        for ensp_id in ensp_ids: # for each ENSP-ID
            gene_exp = get_gene_exp(graph_dict, ensp_id) # get corresponding gene expression in the graph
            gene_exp_list.append(gene_exp) # add gene expression to list
            diseases_score, disgenet_score, malacards_score = get_disease_scores(ensp_id, ensembl_dict_reversed, diseases_score_dict, disgenet_score_dict, malacards_score_dict)
            diseases_score_list.append(diseases_score)
            disgenet_score_list.append(disgenet_score)
            malacards_score_list.append(malacards_score)
            pbar.update(1)
    
    assert(len(gene_exp_list) == df.shape[0])
    df['gene_exp'] = gene_exp_list # add the list to dataframe
    df['disease_score'] = diseases_score_list
    df['disgenet_score'] = disgenet_score_list
    df['malacards_score'] = malacards_score_list 

    df.to_csv(out_dir) # save the csv file


    
    