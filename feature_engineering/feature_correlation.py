"""
Plot heatmap for features of the SNP datasets
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-data_dir',
                        default='../../data/original/NIH_SNPs_DGA_GE.csv',
                        required=False,
                        help='directory of gene expression graph')
                 
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    data_dir = args.data_dir
    df = pd.read_csv(data_dir)
    column_names = [#----non-structual----
                    'wildtype_value', 
                    'mutant_value',    
                    'blosum62_mu', 
                    'blosum62_wt',
                    'entropy', 
                    'diseases_score', 
                    'disgenet_score', 
                    'malacards_score', 
                    'gene_exp',
                    #----structual----
                    'core', 
                    'nis', 
                    'confidence',
                    'interface', 
                    'hbo', 
                    'sbr', 
                    'aromatic', 
                    'hydrophobic', 
                    'helix', 
                    'coil', 
                    'sheet', 
                    #----new-----
                    'wt_calculated_fold_change',
                    'wt_abundance',
                    'mu_calculated_fold_change',
                    'mu_abundance',
                    'ssipe',
                    #----label----
                    'p_value']
    df = df[column_names] #selecting needed columns
    #print(df)

    df_corr = df.corr(method='pearson')
    #print(df_corr)

    fig, ax = plt.subplots()
    im = ax.imshow(df_corr)
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=90)
    ax.set_yticks(range(df.shape[1]))
    ax.set_yticklabels(df.columns)
    ax.xaxis.set_ticks_position('bottom')
    #plt.matshow(df_corr, fignum=f.number)
    
    cb = ax.figure.colorbar(im, ax=ax)
    cb.ax.tick_params(labelsize=10)
    ax.set_title('Correlation Matrix', fontsize=18)
    fig.tight_layout()
    plt.show()