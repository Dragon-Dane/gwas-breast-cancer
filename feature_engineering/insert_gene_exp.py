import argparse
import networkx as nx
import pandas as pd
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-data_dir',
                        default='../../data/original/NIH_SNPs_features.csv',
                        required=False,
                        help='directory of gene expression graph')

    parser.add_argument('-graph_dir',
                        #default='../../data/original/string_ERpos.gexf',
                        default='../../data/original/string_ERpos_neighbours.gexf',
                        required=False,
                        help='directory of the dataset')

    parser.add_argument('-out_dir',
                        default='../../data/output/NIH_SNPs_features_new.csv',
                        required=False,
                        help='directory of output dataset')    
                 
    return parser.parse_args()

def get_gene_exp(graph_dict, ensembl_id):
    """
    Retrieve the gene expression from the graph according to the ensembl id,
    then return the corresponding gene expression as integer.  
    """
    return int(float(graph_dict[ensembl_id]))

if __name__=="__main__":
    args=get_args()
    graph_dir = args.graph_dir
    data_dir = args.data_dir
    out_dir = args.out_dir
    print('------------------------------------------')
    print('graph directory:', graph_dir)
    print('data directory:', data_dir)
    print('output data directory:', out_dir)
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
    df = pd.read_csv(data_dir, delim_whitespace=True)
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
                    'diseases_score', 
                    'disgenet_score', 
                    'malacards_score', 
                    #'gene_exp',
                    'p_value']
    df = df[column_names]
    print('number of samples in original dataset:', df.shape[0])    

    print('entries with -1 p-values:')
    print(df[df.p_value == -1])

    df = df[df.p_value != -1] # exclude the rows which have missing p-values
    print('number of samples after dropping samples with missing p-values:', df.shape[0])

    ensp_ids = list(df['ensembl']) # get ENSP-IDs in the dataset as iterable

    print('Processing the .csv file...')
    gene_exp_list = [] # create an empty list for gene expressions  
    with tqdm(total=df.shape[0]) as pbar:
        for ensp_id in ensp_ids: # for each ENSP-ID
            gene_exp = get_gene_exp(graph_dict, ensp_id) # get corresponding gene expression in the graph
            gene_exp_list.append(gene_exp) # add gene expression to list
            pbar.update(1)
    
    assert(len(gene_exp_list) == df.shape[0])
    df['gene_exp'] = gene_exp_list # add the list to dataframe
    df.to_csv(out_dir) # save the csv file


    
    