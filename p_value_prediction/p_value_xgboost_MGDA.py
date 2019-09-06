'''
Using resgression models to predict p-values according to the features of SNPs.
'''
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import roc_curve
from statistics import mean 
import matplotlib.pyplot as plt

def get_args():
    """
    Argument parsing
    """
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-p_value_threshold',
                        default=0.005,
                        required=False)
    
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
    plt.show()

if __name__ == "__main__":
    args = get_args()
    df = pd.read_csv("../data/final_michailidu_genenames_disease_association.csv", delim_whitespace=True) # read data 
    p_value_threshold = args.p_value_threshold
    print('threshold for p-value is set to ', p_value_threshold)
    
    #------------------------------------------
    #           Data pre-processing
    #------------------------------------------
    column_names = ['wildtype_value', 
                    'mutant_value', 
                    'confidence', 
                    'core', 
                    'NIS',
                    'Interface', 
                    'HBO', 
                    'SBR', 
                    #'Aromatic', 
                    'Hydrophobic',  
                    'Blosum62_mu', 
                    'Blosum62_wt', 
                    'helix', 
                    'coil', 
                    'sheet', 
                    'Entropy', 
                    'malacards_score',
                    'diseases_score', 
                    'disgenet_score', 
                    #'Gene_expression', 
                    'P-value']
    df = df[column_names] #selecting needed columns

    # replace 0 in Gene_expression with np.nan because it represents missing value
    #gene_expressions = df['Gene_expression']
    #gene_expressions = [np.nan if g==0 else g for g in gene_expressions]
    #df['Gene_expression'] = gene_expressions

    # add a column indicating the prediction result (0 or 1)
    p_values = df['P-value']
    p_values_binary = [int(p_value <= p_value_threshold) for p_value in p_values]
    df = df.drop(columns = ['P-value'])
    df['classification_result'] = p_values_binary 

    # the feature "Gene_expression" is a categorical data with 3 classes:
    # [1: down regulated][2:normal expression][3: upregulated]
    # they need to be convert to np.nan to be accurate.

    #calculate class weight
    df_pos = df[df['classification_result'] == 1]
    df_neg = df[df['classification_result'] == 0]
    num_pos = df_pos.shape[0]
    num_neg = df_neg.shape[0]
    print('number of positive samples:', num_pos)
    print('number of negative samples:', num_neg)
    scale_pos_weight = float(num_neg/num_pos)

    # data and label
    X = np.array(df.drop(columns = ['classification_result']))
    y = np.array(df['classification_result'])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    #------------------------------------------
    #           XGBoost classifier
    #------------------------------------------
    # list contains dictionarys for each fold
    train_acc_records = []
    train_precision_records = []
    train_recall_records = []
    val_acc_records = []
    val_precision_records = []
    val_recall_records = []
    val_f1_records = []
    val_mcc_records = []
    fpr_records = []
    tpr_records = []
    thresholds_records = []
    feature_importance_records = []

    # run the model in a cross-validation manner
    print('--------------------------------------------------')
    for train_index, val_index in skf.split(X, y):
        print('Starting cross-validation on XGBoost model...')
        xgb_clf = xgb.XGBClassifier(learning_rate=0.2, n_estimators=3000, booster='gbtree', verbosity=1, n_jobs=1, scale_pos_weight=scale_pos_weight, random_state=123)
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        print('training XGBoost model...')
        xgb_clf.fit(X_train, y_train)# train the random forest
        print('training finished.')

        preds_train = xgb_clf.predict(X_train) # predictions on training data
        preds_val = xgb_clf.predict(X_val) # predictions on validation data

        # training accuracy
        train_acc = accuracy_score(y_train, preds_train)
        train_acc_records.append(train_acc)
        print('training accuracy: ', train_acc)

        # training precision
        train_precision = precision_score(y_train, preds_train)
        train_precision_records.append(train_precision)
        print('training precision:', train_precision)

        # training recall
        train_recall = recall_score(y_train, preds_train)
        train_recall_records.append(train_recall)
        print('training recall: ', train_recall)

        # validation accuracy
        val_acc = accuracy_score(y_val, preds_val)
        val_acc_records.append(val_acc)
        print('validation accuracy: ', val_acc)

        # validation precision
        val_precision = precision_score(y_val, preds_val)
        val_precision_records.append(val_precision)
        print('validation precision:', val_precision)

        # validation recall
        val_recall = recall_score(y_val, preds_val)
        val_recall_records.append(val_recall)
        print('validation recall: ', val_recall)

        # validation F1
        val_f1 = f1_score(y_val, preds_val)
        val_f1_records.append(val_f1)
        print('validation f1:', val_f1)

        # validation MCC
        val_mcc = matthews_corrcoef(y_val, preds_val)
        val_mcc_records.append(val_mcc)
        print('validation MCC:', val_mcc)

        # feature importance 
        feature_importance_records.append(xgb_clf.feature_importances_)
        print('--------------------------------------------------')

    #------------------------------------------
    #           Post-processing 
    #------------------------------------------
    # averaged validation metrics over folds
    print('averaged training accuracy:', mean(train_acc_records))
    print('averaged training precision:', mean(train_precision_records))
    print('averaged training recall:', mean(train_recall_records))
    print('averaged validation accuracy:', mean(val_acc_records))
    print('averaged validation precision:', mean(val_precision_records))
    print('averaged validation recall:', mean(val_recall_records))
    print('averaged validation F1:', mean(val_f1_records))
    print('averaged validation MCC:', mean(val_mcc_records))

    # rank the features
    feature_rank(feature_importance_records, column_names)