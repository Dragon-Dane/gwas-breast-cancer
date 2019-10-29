"""
Run t-sne algorithm on the patient dataset to reduce the dimensionality
"""
import argparse
import pandas as pd 
import numpy as np 
from sklearn.decomposition import TruncatedSVD

def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-data_dir',
                        default='../../data/patient_output/patients_all.csv',
                        required=False,
                        help='directory of SNP dataframe')

    parser.add_argument('-out_dir',
                        default='../../data/patient_output/patients_latent_truncated_svd_256.csv',
                        required=False,
                        help='csv file containing all features of snps of the patients.')
                    
    return parser.parse_args()

if __name__=="__main__":
    args = get_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    df = pd.read_csv(data_dir)
    X = np.array(df.drop(columns = ['class'])) # drop the result column
    y = df['class']
    print('shape of input features:', X.shape)
    
    
    # perform T-SNE
    svd = TruncatedSVD(n_components=256)
    print('begin to fit...')
    X_out = svd.fit_transform(X)
    print('training finished.')
    print('explained_variance_ratio:')
    print(svd.explained_variance_ratio_)
    print('total explained variance_ratio')  
    print(svd.explained_variance_ratio_.sum())  
    print('shape of output features:', X_out.shape)
    df_out = pd.DataFrame(X_out)
    df_out['class'] = y
    print('shape of output dataframe: ', df_out.shape)
    df_out.to_csv(out_dir)
    print('program finished.')



