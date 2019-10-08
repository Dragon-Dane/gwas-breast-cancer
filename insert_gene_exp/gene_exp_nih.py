import argparse
import networkx as nx
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-data_dir',
                        default='../../data/original/NIH_SNPs_features.csv',
                        required=False,
                        help='directory of gene expression graph')

    parser.add_argument('-graph_dir',
                        default='../../data/original/string_ERpos.gexf',
                        required=False,
                        help='directory of the dataset')

    parser.add_argument('-out_dir',
                        default='../../data/output/NIH_SNPs_features_new.csv',
                        required=False,
                        help='directory of output dataset')    
                 
    return parser.parse_args()

if __name__=="__main__":
    args=get_args()
    graph_dir = args.graph_dir
    data_dir = args.data_dir
    out_dir = args.out_dir

    # read the graph
    #gene_exp = nx.read_gexf(graph_dir) # read gene expressions as graph
    #protein_labels = nx.get_node_attributes(gene_exp, "ENSP-ID")
    #print(protein_labels)

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

    # exclude the rows which have missing p-values
    df = df[df.p_value != -1]
    print('number of samples after dropping samples with missing p-values:', df.shape[0])

    # get ENSP-IDs as iterable
    ensp_ids = list(df['ensembl'])
    #print(ensp_ids)

    # create an empty list for gene expressions
    gene_exp_list = []
    # for each ENSP-ID
    for ensp_id in ensp_ids:
        print(ensp_id)
        # get corresponding gene expression in the graph
        # add gene expression to list
    # add the list to dataframe
    # save the csv file
    
    