'''
ROC plot of a dataset
'''
import argparse
import pickle
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


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

    result_dir = './results/roc_patient_classification.pickle'

    result_file = open(result_dir, 'rb')
    roc_list = pickle.load(result_file)

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr, mean_auc = roc_process(roc_list, mean_fpr)

    # plot roc curve of random
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random', alpha=.8)
    
    plt.plot(mean_fpr, mean_tpr, dashes = [6, 1, 1, 1], color='b', label='Random Forest', lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()