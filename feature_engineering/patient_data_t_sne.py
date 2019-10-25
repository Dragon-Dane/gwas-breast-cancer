"""
Perform t-sne on the patient dataset, then store the new data set in a dataframe in a csv file.
"""
import argparse
import pandas as pd 
import numpy as np 
import json
from os import listdir
from os.path import isfile, join
import multiprocessing

def get_args():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-op',
                        default='integrate',
                        required=False,
                        choices=['integrate', 't-sne'],
                        help='Wether to integrate patient data or run t-sne one the integrated data.')

    parser.add_argument('-snp_dir',
                        default='../../data/original/NIH_SNPs_DGA_GE.csv',
                        required=False,
                        help='directory of SNP dataframe')

    parser.add_argument('-patient_root_dir',
                        default='../../data/patient-original/patient_genotypes/',
                        required=False,
                        help='root directory of the patients')

    parser.add_argument('-num_latent_features',
                         default = 128,
                         required=False)

    parser.add_argument('-out_dir',
                        default='../../data/patient_output/patients_1.csv',
                        required=False,
                        help='csv file containing all features of snps of the patients.')
                    
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
    #print(snp_df)
    return snp_df


def patient_list_gen(patient_root_dir):
    """
    Returns list of csv files of patients. 
    """
    patient_list = listdir(patient_root_dir)
    #patient_list.sort()
    return patient_list


def patient_dir_gen(patient_root_dir, patient_file):
    """
    Returns the directory of a patient file.
    """
    patient_dir = patient_root_dir + patient_file
    return patient_dir

def unavail_snps():
    """
    Returns a list of unavailable SNPs that don't have info in the dataset. 
    """
    with open('./snp_without_features.json') as snp_without_features:
        data = json.load(snp_without_features)
        snp_without_features.close()
        return data


def patient_df_gen(patient_root_dir, patient_file, snp_df, snps_without_features, features_to_mask):
    """
    Returns an falttened dataframe of a single patient. The attributes 
    of the SNPs are added to the original patient dataframe.
    snps_without_features: list of SNPs which are not available in snp_df
    """
    patient_dir = patient_dir_gen(patient_root_dir, patient_file)

    patient_df =  pd.read_csv(patient_dir) # dataframe of a patient
    patient_label = patient_df.iloc[0, -1] # sick or not
    patient_age = patient_df.iloc[0, -2] # age
    patient_df = patient_df.drop(['Age', 'Class'], axis=1)
    patient_df = patient_df.drop(snps_without_features, axis=1)
    snp_list = list(patient_df.columns) # list of snps in the patient (which are also available in the SNP dataset)
    #print('length of the snp_list of patient:', len(snp_list))

    patient_df_new = pd.DataFrame()
    for snp in snp_list:
        geno_type = patient_df.iloc[0][snp] # geno_type of the SNP in the patient
        snp_features = snp_df.loc[[snp]] # get corresponding features of that SNP
        #if snp_features.shape[0] != 1:
        #    print(snp_features.shape)
        #    print(snp)
        #print(snp_features)
        if geno_type == 1 or geno_type == 2: # keep everything still
            patient_df_new = patient_df_new.append(snp_features)
        elif geno_type == 0 or geno_type == -1: # mask everything to 0 except disease scores
            snp_features[features_to_mask] = 0 
            patient_df_new = patient_df_new.append(snp_features)
        else:
            raise ValueError('invalid geno_type {} for snp {}'.format(geno_type, snp))
    #print('shape of new patient dataframe:', patient_df_new.shape)

    patient_df_new = patient_df_new.unstack().to_frame().sort_index(level=1).T # flatten the dataframe to one row
    patient_df_new.columns = patient_df_new.columns.map('_'.join) # flatten the dataframe to one row
    patient_df_new['age'] = patient_age
    patient_df_new['class'] = patient_label

    #print('shape of new patient dataframe:', patient_df_new.shape)
    return patient_df_new

def tup_gen_for_multiprocs(patient_root_dir, patient_files, snp_df, snps_without_features, features_to_mask):
    """
    Generate a list of tuples containing all the input combinations of patient_df_gen.
    """
    tups = [(patient_root_dir, patient_file, snp_df, snps_without_features, features_to_mask) for patient_file in patient_files]
    return tups

if __name__ == "__main__":
    args = get_args()
    snp_dir = args.snp_dir
    patient_root_dir = args.patient_root_dir
    num_latent_features = args.num_latent_features
    out_dir = args.out_dir
    # Fetures to set to 0 when the corresponding genotype is -1 or 0.
    features_to_mask = [#----non-structual----
                    'wildtype_value', 
                    'mutant_value',    
                    'blosum62_mu', 
                    'blosum62_wt',
                    'entropy', 
                    #'diseases_score', 
                    #'disgenet_score', 
                    #'malacards_score', 
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

    snp_df = snp_df_gen(snp_dir) # dataframe of SNPs indexed by SNP-id
    print('shape of SNPs dataframe: ', snp_df.shape)

    patient_list = patient_list_gen(patient_root_dir) # list of files of patients
    print('total number of patients: ', len(patient_list))
    #patient_files = [patient_list[0],patient_list[1], patient_list[2]] 
    #print(patient_files)
    snps_without_features = unavail_snps()# SNPs without features

    #total_df = pd.DataFrame() # the dataframe to store all the attributes of all the patients
    #for patient_file in patient_files:
    #    patient_new_df = patient_df_gen(patient_root_dir, patient_file, snp_df, snps_without_features, features_to_mask) 
    #    total_df = total_df.append(patient_new_df)
    
    tups = tup_gen_for_multiprocs(patient_root_dir, patient_list, snp_df, snps_without_features, features_to_mask)
    with multiprocessing.Pool() as pool:
        total_df = pool.starmap(patient_df_gen, tups)
    total_df = pd.concat(total_df)
    print(total_df)
    total_df.to_csv(out_dir)
    print('program finished.')
