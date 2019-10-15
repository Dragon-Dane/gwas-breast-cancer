'''
Perform 5-fold cross validation and show the classification performance for
pre-trained random forest classifiers. The pre-trained models are trained by
random_forest_tuning.py
'''
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix
from statistics import mean 
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-dataset',
                        default = 'nih',
                        required = False,
                        choices = ['nih', 'nih_nodup', 'erneg', 'erpos'])

    parser.add_argument('-use_gene_expression',
                         default = 'True',
                         required = False,
                         choices = ['True', 'False'])

    parser.add_argument('-p_value_th',
                        default = 0.05,
                        required = False)
    return parser.parse_args()

def feature_rank(feature_importance_records, column_names):
    """
    Visualize the feature importances.
    feature_importance_records: list containing feature importances from
    5-fold cross-validation
    """
    feature_importance_records = np.array(feature_importance_records)
    avg_feature_importance = np.mean(feature_importance_records, axis=0)
    column_names = column_names[:-1]

    # sort the feature importance
    feature_importance_with_names = list(zip(column_names,avg_feature_importance))
    feature_importance_with_names.sort(key = lambda tup: tup[1], reverse=True)
    #print(feature_importance_with_names)
    avg_feature_importance = [tup[1] for tup in feature_importance_with_names]
    column_names = [tup[0] for tup in feature_importance_with_names]
    #print(avg_feature_importance)
    #print(column_names)

    x_pos = [i for i, _ in enumerate(avg_feature_importance)]
    #print(avg_feature_importance.shape, len(column_names), x_pos)
    plt.figure(figsize=(16,8))
    plt.bar(x_pos, avg_feature_importance, color='lightseagreen')
    plt.xlabel("Feature name")
    plt.ylabel("importance")
    plt.title("Feature importances")
    plt.xticks(x_pos, column_names, rotation=35, ha='right')

if __name__ == "__main__":
    args = get_args()
    dataset = args.dataset
    use_gene_expression = args.use_gene_expression
    p_value_threshold = args.p_value_th
    print('dataset:', dataset)
    print('threshold for p-value is set to ', p_value_threshold)
    #------------------------------------------
    #           Data pre-processing
    #------------------------------------------
    if dataset == 'nih':
        #df = pd.read_csv("../../data/output/NIH_SNPs_features_new.csv", delim_whitespace=True)
        df = pd.read_csv("../../data/output/NIH_SNPs_features_new.csv")
        best_param_dir = './best_param/random_forest_nih.pickle'
    elif dataset == 'erneg':
        df = pd.read_csv("../../data/output/michailidu_SNPs_features_ERneg_new.csv")
        best_param_dir = './best_param/michailidu_erneg.pickle' 
    elif dataset == 'erpos':
        df = pd.read_csv("../../data/output/michailidu_SNPs_features_ERpos_new.csv")
        best_param_dir = './best_param/michailidu_erpos.pickle'
    elif dataset == 'nih_nodup':
        df = pd.read_csv("../../data/output/NIH_part_SNPs_features_nodup_new.csv")
        best_param_dir = './best_param/random_forest_nih_nodup.pickle'          
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
    df = df[column_names] #selecting needed columns
    if use_gene_expression == 'False':
        df = df.drop(columns = ['gene_exp'])

    # add a column indicating the prediction result (0 or 1)
    p_values = df['p_value']
    p_values = np.exp([p_value * (-1) for p_value in p_values]) # convert to original values from -log values  
    p_values_binary = [int(p_value <= p_value_threshold) for p_value in p_values]
    df = df.drop(columns = ['p_value'])
    df['classification_result'] = p_values_binary 
    print(df)

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
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    #print(X.shape)
    #print(y.shape)

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
    feature_importance_records = []

    # load pre-selected hyper parameters
    best_param_file = open(best_param_dir, 'rb')
    best_param = pickle.load(best_param_file)
    print('hyper-parameters selected from randomized search:')
    print(best_param)

    # run the model in a cross-validation manner
    for train_index, val_index in skf.split(X, y):

        # load the hyper-parameters into the random forest model
        rf = RandomForestClassifier(n_estimators=best_param['n_estimators'],
                                    criterion=best_param['criterion'],
                                    max_features=best_param['max_features'],
                                    max_depth=best_param['max_depth'],
                                    min_samples_split=best_param['min_samples_split'],
                                    min_samples_leaf=best_param['min_samples_leaf'], 
                                    bootstrap=best_param['bootstrap'],
                                    random_state=123,
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

        feature_importance_records.append(rf.feature_importances_)

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
    feature_rank(feature_importance_records, column_names)
    plt.savefig('./figures/random_forest_' + dataset + '_use_gene_expression_' + use_gene_expression + '.png')
    plt.show()