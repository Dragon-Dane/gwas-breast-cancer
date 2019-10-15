"""
This code is used to find the optimal random forest hyper-parameters for each dataset.
Basically, it does a grid search on the specified hyper-parameters.
"""
import argparse
import numpy as np  
import pandas as pd  
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from time import time
import pickle
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

    parser.add_argument('-seed',
                        default = 42,
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
    seed = args.seed
    print('dataset:', dataset)
    print('use gene expression feature:', use_gene_expression)
    #-----------------------------------------------------------
    #              Data pre-processing
    #-----------------------------------------------------------
    if dataset == 'nih':
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
    df = df[column_names]
    if use_gene_expression == 'False':
        df = df.drop(columns = ['gene_exp'])

    # data and label
    X = np.array(df.drop(columns = ['p_value']))
    y = np.array(df['p_value'])
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    #-----------------------------------------------------------
    #                   Random forest
    #-----------------------------------------------------------
    train_mse_records = []
    train_r2_records = []
    val_mse_records = []
    val_r2_records = []
    feature_importance_records = []

    # load pre-selected hyper parameters
    best_param_file = open(best_param_dir, 'rb')
    best_param = pickle.load(best_param_file)
    print('hyper-parameters selected from randomized search:')
    print(best_param)
    
    # run the model in a cross-validation manner
    for train_index, val_index in kf.split(X):

        # load the hyper-parameters into the random forest model
        rf = RandomForestRegressor(n_estimators=best_param['n_estimators'],
                                    criterion=best_param['criterion'],
                                    max_features=best_param['max_features'],
                                    max_depth=best_param['max_depth'],
                                    min_samples_split=best_param['min_samples_split'],
                                    min_samples_leaf=best_param['min_samples_leaf'], 
                                    bootstrap=best_param['bootstrap'],
                                    random_state=seed,
                                    n_jobs = -1)

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # train the rf on the training data
        print('training the random forest...')
        rf.fit(X_train, y_train)
        print('training finished.')

        y_train_pred = rf.predict(X_train)
        y_val_pred = rf.predict(X_val)

        mse_train = mean_squared_error(y_train, y_train_pred)
        train_mse_records.append(mse_train)
        print('mean squared error for training: ', mse_train)

        r2_train = r2_score(y_train, y_train_pred)
        train_r2_records.append(r2_train)
        print('r2 score for training: ', r2_train)

        mse_val = mean_squared_error(y_val, y_val_pred)
        val_mse_records.append(mse_val)
        print('mean squared error for validation: ', mse_val)

        r2_val = r2_score(y_val, y_val_pred)
        val_r2_records.append(r2_val)
        print('r2 score for validation:', r2_val)

        feature_importance_records.append(rf.feature_importances_)
        print('--------------------------------------------------')

    #------------------------------------------
    #           Post-processing 
    #------------------------------------------
    # averaged validation metrics over folds
    print('averaged training mse:', mean(train_mse_records))
    print('averaged training r2 score:', mean(train_r2_records))
    print('averaged validation mse:', mean(val_mse_records))
    print('averaged validation r2 score:', mean(val_r2_records))

    # rank the features
    feature_rank(feature_importance_records, column_names)
    plt.savefig('./figures/random_forest_' + dataset + '_use_gene_expression_' + use_gene_expression + '.png')
    plt.show()