import argparse
import networkx as nx
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-data_dir',
                        default='../../GWAS-data/genexp_nanvalue_changement.gexf',
                        required=False,
                        help='Operation mode, control_vs_heme,control_vs_nucleotide')

    parser.add_argument('-out_dir',
                        default='../../GWAS-data/output/',
                        required=False,
                        help='Operation mode, control_vs_heme,control_vs_nucleotide')    
                 
    return parser.parse_args()

if __name__=="__main__":
    column_names = ['wildtype_value', 
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
                    'gene_exp',
                    'p_value']

    args=get_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    G = nx.read_gexf(data_dir)
    protein_labels = nx.get_node_attributes(G, "ENSP-ID")
    print(protein_labels)
    