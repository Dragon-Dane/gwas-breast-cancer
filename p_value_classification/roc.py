'''
ROC plot of a dataset
'''
import argparse
import pickle
import numpy as np

def get_args():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-dataset',
                        default = 'nih',
                        required = False,
                        choices = ['nih', 'nih_nodup', 'erneg', 'erpos'])
    return parser.parse_args()

def roc_process(roc_dict):
    '''
    Average the data over folds
    '''
    fpr_records = roc_dict['fpr_records']
    tpr_records = roc_dict['tpr_records']
    fpr_records = np.array([np.array(l) for l in fpr_records])
    tpr_records = np.array([np.array(l) for l in tpr_records])
    print(fpr_records)
    #print(tpr_records)

if __name__=="__main__":
    args = get_args()
    dataset = args.dataset
    if dataset == 'nih':
        result_dir_0 = './results/roc_nih_use_structual_features_True.pickle'
        result_dir_1 = './results/roc_nih_use_structual_features_False.pickle'
    elif dataset == 'erneg':
        result_dir_0 = './results/roc_erneg_use_structual_features_True.pickle'
        result_dir_1 = './results/roc_erneg_use_structual_features_False.pickle'
    elif dataset == 'erpos':
        result_dir_0 = './results/roc_erpos_use_structual_features_True.pickle'
        result_dir_1 = './results/roc_erpos_use_structual_features_False.pickle'
    elif dataset == 'nih_nodup':
        result_dir_0 = './results/roc_nih_nodup_use_structual_features_True.pickle'
        result_dir_1 = './results/roc_nih_nodup_use_structual_features_False.pickle'   

    result_file_0 = open(result_dir_0, 'rb')
    roc_dict_0 = pickle.load(result_file_0)    
    result_file_1 = open(result_dir_1, 'rb')
    roc_dict_1 = pickle.load(result_file_1)
    roc_process(roc_dict_0)

