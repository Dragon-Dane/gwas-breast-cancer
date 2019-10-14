import argparse
import pandas as pd
#from tqdm import tqdm
def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-data_dir',
                        default='../../data/original/NIH_part_SNPs_features_nodup.csv',
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
                 
if __name__=="__main__":
    args=get_args()
    data_dir = args.data_dir
    out_dir = args.out_dir

    # read the dataframe from csv file without gene expressions
    df = pd.read_csv(data_dir)
    #df = pd.read_csv(data_dir, delim_whitespace=True)
    print(df)

    # combine 'struct_pos' feature
    column_names = ['core','nis','interface']
    df_struct_pos = df[column_names]
    print(df_struct_pos)

    # combine 'sec_struct' feature
    