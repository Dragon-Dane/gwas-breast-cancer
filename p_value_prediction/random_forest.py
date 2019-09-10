'''
Using resgression models to predict p-values according to the features of SNPs.
'''
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix
from statistics import mean 

def get_args():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-dataset',
                        default = 'NIH',
                        required = False,
                        choices = ['NIH', 'ERneg', 'ERpos'])
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    dataset = args.dataset
    print('dataset:', dataset)

    #------------------------------------------
    #           Data pre-processing
    #------------------------------------------
    p_value_threshold = 0.05
    print('threshold for p-value is set to ', p_value_threshold)

    if dataset == 'NIH':
        df = pd.read_csv("../data/NIH_SNPs_features.csv", delim_whitespace=True)
    elif dataset == 'ERneg':
        df = pd.read_csv("../data/michailidu_SNPs_features_ERneg.csv", delim_whitespace=True) 
    elif dataset == 'ERpos':
        df = pd.read_csv("../data/michailidu_SNPs_features_ERpos.csv", delim_whitespace=True) 

    df = df[['wildtype_value', 
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
             'p_value' 
             #'gene_exp'
             ]] #selecting needed columns
    print(df)

    # add a column indicating the prediction result (0 or 1)
    p_values = df['p_value']
    p_values = np.exp([p_value * (-1) for p_value in p_values]) # convert to original values from -log values  
    p_values_binary = [int(p_value <= 0.05) for p_value in p_values]
    df = df.drop(columns = ['p_value'])
    df['classification_result'] = p_values_binary 
    print(df)

    # the feature "Gene_expression" is a categorical data with 3 classes:
    # [1: down regulated][2:normal expression][3: upregulated]
    # they need to be convert to one-hot to be well used by the classifier because they are
    # actually not ordinal.

    #calculate class weight
    df_pos = df[df['classification_result'] == 1]
    df_neg = df[df['classification_result'] == 0]
    num_pos = df_pos.shape[0]
    num_neg = df_neg.shape[0]
    print('number of positive samples:', num_pos)
    print('number of negative samples:', num_neg)
    class_weight = {0:num_pos, 1:num_neg} 

    # data and label
    X = np.array(df.drop(columns = ['classification_result']))
    y = np.array(df['classification_result'])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    #------------------------------------------
    #           Random Forest 
    #------------------------------------------
    # list contains dictionarys for each fold
    train_acc_records = []
    val_acc_records = []
    val_precision_records = []
    val_recall_records = []
    val_mcc_records = []
    fpr_records = []
    tpr_records = []
    thresholds_records = []

    # run the model in a cross-validation manner
    for train_index, val_index in skf.split(X, y):

        # seperate hyper-parameters for different datasets
        if dataset == 'NIH':
            rf = RandomForestClassifier(n_estimators=1000,
                                        max_depth=None,
                                        min_samples_split=5,
                                        random_state=0,
                                        max_features = "auto",
                                        criterion = "gini",
                                        class_weight = class_weight,
                                        n_jobs = -1)
        elif dataset == 'ERneg':
            rf = RandomForestClassifier(n_estimators=1400,
                                        max_depth=None,
                                        min_samples_split=5,
                                        random_state=0,
                                        max_features = "auto",
                                        criterion = "gini",
                                        class_weight = class_weight,
                                        n_jobs = -1)            
        elif dataset == 'ERpos':
            rf = RandomForestClassifier(n_estimators=1000,
                                        max_depth=None,
                                        min_samples_split=5,
                                        random_state=0,
                                        max_features = "auto",
                                        criterion = "gini",
                                        class_weight = class_weight,
                                        n_jobs = -1)

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        rf.fit(X_train, y_train)# train the random forest

        # train the rf on the training data
        print('training the random forest...')
        rf.fit(X_train, y_train)
        print('training finished.')

        # training performance
        train_acc = rf.score(X_train, y_train)
        train_acc_records.append(train_acc)
        print('training accuracy:', train_acc)

        # validation performance
        predictions = rf.predict(X_val)
        num_correct = np.sum(predictions == y_val)
        print('number of correct predictions:',num_correct)

        val_acc = rf.score(X_val, y_val)
        val_acc_records.append(val_acc)
        print('validation accuracy:', val_acc)

        val_precision = precision_score(y_val,predictions)
        val_precision_records.append(val_precision)        
        print('validation precision:', val_precision)

        val_recall = recall_score(y_val,predictions)
        val_recall_records.append(val_recall)        
        print('validation recall:', val_recall)

        val_mcc = matthews_corrcoef(y_val, predictions)
        val_mcc_records.append(val_mcc)
        print('validation mcc:', val_mcc)

        val_f1 = f1_score(y_val, predictions)
        print('validation f1:', val_f1)

        val_cm = confusion_matrix(y_val, predictions)
        print('validation confusion matrix:', val_cm)

        # output probabilities for val data
        val_prob = rf.predict_proba(X_val)
        fpr, tpr, thresholds = roc_curve(y_val, val_prob[:, 1])

        # convert to list so that can be saved in .json file
        fpr = fpr.tolist()
        tpr = tpr.tolist()
        thresholds = thresholds.tolist()
        fpr_records.append(fpr)
        tpr_records.append(tpr)
        thresholds_records.append(thresholds)
        print('--------------------------------------------------')

    #------------------------------------------
    #           Post-processing 
    #------------------------------------------
    # averaged validation metrics over folds
    print('averaged training accuracy:', mean(train_acc_records))
    print('averaged validation accuracy:', mean(val_acc_records))
    print('averaged validation precision:', mean(val_precision_records))
    print('averaged validation recall:', mean(val_recall_records))
    print('averaged validation MCC:', mean(val_mcc_records))

    # rank the features