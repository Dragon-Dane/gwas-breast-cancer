'''
ROC plot of a dataset
'''
import argparse
import pickle
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def get_args():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-dataset',
                        default = 'nih',
                        required = False,
                        choices = ['nih', 'nih_nodup', 'erneg', 'erpos'])
    return parser.parse_args()

def roc_process(roc_list, mean_fpr):
    '''
    Average the data over folds
    '''
    tprs = []
    aucs = []    
    i = 0
    for roc in roc_list:
        fpr = roc['fpr']
        tpr = roc['tpr']
        auc_value = roc['auc']
        tprs.append(interp(mean_fpr, fpr, tpr))
        aucs.append(auc_value)
        tprs[-1][0] = 0.0
        #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, auc))
        i+=1
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    return mean_tpr, mean_auc    


if __name__=="__main__":
    args = get_args()
    dataset = args.dataset
    if dataset == 'nih':
        result_dir_0 = './results/roc_nih_use_structual_features_True_p_value_0,05.pickle'
        result_dir_1 = './results/roc_nih_use_structual_features_False_p_value_0,05.pickle'
    elif dataset == 'erneg':
        result_dir_0 = './results/roc_erneg_use_structual_features_True_p_value_0,05.pickle'
        result_dir_1 = './results/roc_erneg_use_structual_features_False_p_value_0,05.pickle'
    elif dataset == 'erpos':
        result_dir_0 = './results/roc_erpos_use_structual_features_True_p_value_0,05.pickle'
        result_dir_1 = './results/roc_erpos_use_structual_features_False_p_value_0,05.pickle'
    elif dataset == 'nih_nodup':
        result_dir_0 = './results/roc_nih_nodup_use_structual_features_True_p_value_0,05.pickle'
        result_dir_1 = './results/roc_nih_nodup_use_structual_features_False_p_value_0,05.pickle'   

    result_file_0 = open(result_dir_0, 'rb')
    roc_list_0 = pickle.load(result_file_0)    
    result_file_1 = open(result_dir_1, 'rb')
    roc_list_1 = pickle.load(result_file_1)
    
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr_0, mean_auc_0 = roc_process(roc_list_0, mean_fpr)
    mean_tpr_1, mean_auc_1 = roc_process(roc_list_1, mean_fpr)
    mean_fpr = np.linspace(0, 1, 100)

    # plot roc curve of random
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)
    
    
    plt.plot(mean_fpr, mean_tpr_0, color='b', label=r'Mean ROC (AUC = %0.3f)' % (mean_auc_0), lw=2, alpha=.8)
    plt.plot(mean_fpr, mean_tpr_1, color='g', label=r'Mean ROC (AUC = %0.3f)' % (mean_auc_1), lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()