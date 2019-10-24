"""
Perform t-sne on the patient dataset, then store the new data set in a dataframe in a csv file.
"""
import argparse
import pandas as pd 
import numpy as np 
from os import listdir
from os.path import isfile, join

def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-snp_dir',
                        default='../../data/original/NIH_SNPs_DGA_GE.csv',
                        required=False,
                        help='directory of SNP dataframe')

    parser.add_argument('-patient_root_dir',
                        default='../../data/patient-original/patient_genotypes/',
                        required=False,
                        help='root directory of the patients')
                    
    return parser.parse_args()

def snp_df_gen(snp_dir):
    """
    Returns a SNP dataframe, the snp ids are used as indexes
    """
    df = pd.read_csv(snp_dir)
    selected_features = ['snp_id',
                         #----non-structual----
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
    df = df[selected_features]
    snp_df = df.set_index('snp_id')
    print(snp_df)
    return snp_df


def patient_list_gen(patient_root_dir):
    """
    Returns list of csv files of patients. 
    """
    patient_list = listdir(patient_root_dir)
    return patient_list


def patient_dir_gen(patient_root_dir, patient_file):
    patient_dir = patient_root_dir + patient_file
    return patient_dir


def patient_df_gen(patient_dir, snp_df):
    """
    Returns an falttened dataframe of a single patient. The attributes 
    of the SNPs are added to the original patient dataframe.
    """
    patient_df =  pd.read_csv(patient_dir)
    print(patient_df)
    patient_label = patient_df.iloc[0, -1]
    patient_age = patient_df.iloc[0, -2]
    print(patient_label)    
    print(patient_age)
    patient_df = patient_df.drop(['Age', 'Class'], axis=1)
    print(patient_df.shape)
    
    patient_df_new = pd.DataFrame()
    snp_list = list(patient_df.columns)
    for snp in snp_list:
        print(snp)
        

    #patient_df_new
    return patient_df_new


if __name__ == "__main__":
    args = get_args()
    snp_dir = args.snp_dir
    patient_root_dir = args.patient_root_dir

    snp_df = snp_df_gen(snp_dir) # dataframe of SNPs indexed by SNP-id

    patient_list = patient_list_gen(patient_root_dir) # list of files of patients
    print('total number of patients: ', patient_list)

    patient_file = patient_list[0]
    patient_dir = patient_dir_gen(patient_root_dir, patient_file)
    patient_df_gen(patient_dir, snp_df)
    
    # Put all the patient's attributes in a large matirx. Could be very large, use bigmem nodes. 
    #for i in range(m):

    # Labels of patients
    
    # Random forest
