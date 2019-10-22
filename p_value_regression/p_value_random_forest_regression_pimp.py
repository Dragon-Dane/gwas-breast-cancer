"""
Random forest for p-value prediction. The feature ranking is done by permutation importance.
"""
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from statistics import mean 
import matplotlib.pyplot as plt
from time import time
from rfpimp import *
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-dataset',
                        default = 'erpos',
                        required = False,
                        choices = ['nih_nodup', 'erneg', 'erpos'])

    parser.add_argument('-use_structual_features',
                         default = 'True',
                         required = False,
                         choices = ['True', 'False'])

    parser.add_argument('-grouping',
                         default = 'all_5_groups',
                         required = False,
                         choices = ['all_individual', 'non_struct_individual', 'all_5_groups', 'all_3_groups'])

    parser.add_argument('-show_std',
                         default = 'False',
                         required = False,
                         choices = ['True', 'False'])

    parser.add_argument('-seed',
                        default = 42,
                        required = False)
    return parser.parse_args()

def np_array_to_df(data, column_names):
    #print(data.shape)
    #print(column_names)
    df=pd.DataFrame(data=data[0:,0:], index=[i for i in range(data.shape[0])], columns=column_names)
    return df

def avg_importance(importance_list):
    '''
    Average the feature importances from cross validation
    '''
    df = pd.DataFrame()
    i = 0
    for imp in importance_list:
        #print(imp)
        imp = imp.sort_values(by=['Feature'])
        if i == 0: # get the names of features
            features_list = [row for row in imp.index]
            df['feature'] = features_list
        df['importance_' + str(i)] = list(imp['Importance'])
        #print(df)
        i += 1
    print(df)

    # average the importances in the dataframe
    df1 = df[['importance_' + str(i) for i in range(5)]]
    print(df1)
    df['avg_importance'] = df1.mean(axis=1)
    df['std_importance'] = df1.std(axis=1)
    print(df)

    # get a new data frame of features and averaged importance
    df2 = df[['feature', 'avg_importance', 'std_importance']]
    df2 = df2.sort_values(by=['avg_importance'], ascending=False) # sort by value
    plot_features = list(df2['feature'])
    plot_importances = list(df2['avg_importance'])
    plot_std = list(df2['std_importance'])
    print(df2)
    print(plot_features)
    print(plot_importances)
    print(plot_std)
    return plot_features, plot_importances, plot_std  

def scatter_gen(y_vals, y_preds):
    """
    Generate data for scatter plot of the labels and predicted values
    """
    assert len(y_vals) == len(y_preds)
    tup_list = [(y_vals[i], y_preds[i]) for i in range(0, len(y_vals))]
    tup_list = sorted(tup_list, key = lambda x: x[0], reverse=False) # sort according to value of labels
    x = [tup[0] for tup in tup_list]
    y = [tup[1] for tup in tup_list]
    return x, y


def gen_grouping(option):
    """
    Return lists of features grouped in different ways. Grouped features are stored in 
    sub-lists in the main list.
    """
    feature_separate_all = [#----non-structual----
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
                            'confidence',
                            'core', 
                            'nis', 
                            'interface', 
                            'hbo', 
                            'sbr', 
                            'aromatic', 
                            'hydrophobic', 
                            'helix', 
                            'coil', 
                            'sheet']
    feature_separate_non_structural = [#----non-structural----
                                       'wildtype_value', 
                                       'mutant_value',    
                                       'blosum62_mu', 
                                       'blosum62_wt',
                                       'entropy', 
                                       'diseases_score', 
                                       'disgenet_score', 
                                       'malacards_score', 
                                       'gene_exp']
    feature_group_5 = [#----non-structural----
                        ['wildtype_value', 'mutant_value', 'blosum62_mu', 'blosum62_wt', 'entropy'], # sequence 
                        ['diseases_score', 'disgenet_score', 'malacards_score', 'gene_exp'], # others 
                        ['core', 'nis', 'interface','confidence'], # structure
                        ['hbo', 'sbr', 'aromatic', 'hydrophobic'], # interactions 
                        ['helix', 'coil', 'sheet'] # secondary structure 
                      ]
    feature_group_3 = [#----non-structural----
                        ['wildtype_value', 'mutant_value', 'blosum62_mu', 'blosum62_wt', 'entropy'], # sequence 
                        ['diseases_score', 'disgenet_score', 'malacards_score', 'gene_exp'], # others 
                        ['core', 'nis', 'interface','confidence', 'hbo', 'sbr', 'aromatic', 'hydrophobic', 'helix', 'coil', 'sheet'] # structure
                      ]

    if option == 'all_individual':
        return feature_separate_all
    elif option == 'non_struct_individual':
        return feature_separate_non_structural
    elif option == 'all_5_groups':
        return feature_group_5
    elif option == 'all_3_groups':
        return feature_group_3
    else:
        raise ValueError('Invalid options, choices: [all_individual, non_struct_individual, all_5_groups, all_3_groups]')

if __name__ == "__main__":
    args = get_args()
    dataset = args.dataset
    grouping = args.grouping
    use_structual_features = args.use_structual_features
    show_std = args.show_std
    seed = args.seed
    print('dataset:', dataset)
    print('Using structural features:', use_structual_features)
    grouping_list = gen_grouping(grouping)
    
    #------------------------------------------
    #           Data pre-processing
    #------------------------------------------
    if dataset == 'erneg':
        df = pd.read_csv("../../data/output/michailidu_SNPs_features_ERneg_new.csv")
        best_param_dir = './best_param/michailidu_erneg.pickle'
    elif dataset == 'erpos':
        df = pd.read_csv("../../data/output/michailidu_SNPs_features_ERpos_new.csv")
        best_param_dir = './best_param/michailidu_erpos.pickle'
    elif dataset == 'nih_nodup':
        df = pd.read_csv("../../data/output/NIH_part_SNPs_features_nodup_new.csv")
        best_param_dir = './best_param/random_forest_nih_nodup.pickle'      
    column_names = [#----non-structual----
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
                    'confidence',
                    'core', 
                    'nis', 
                    'interface', 
                    'hbo', 
                    'sbr', 
                    'aromatic', 
                    'hydrophobic', 
                    'helix', 
                    'coil', 
                    'sheet', 
                    #----label----
                    'p_value']
    if use_structual_features == 'False':
        column_names = [#----non-structual----
                        'wildtype_value', 
                        'mutant_value',    
                        'blosum62_mu', 
                        'blosum62_wt',
                        'entropy', 
                        'diseases_score', 
                        'disgenet_score', 
                        'malacards_score', 
                        'gene_exp',
                        #----label----
                        'p_value']
    df = df[column_names] #selecting needed columns

    p_values = df['p_value']
    p_values = np.exp([p_value * (-1) for p_value in p_values]) # convert to original values from -log values  
    df = df.drop(columns = ['p_value'])
    df['p_value'] = p_values 

    # data and label
    X = np.array(df.drop(columns = ['p_value']))
    y = np.array(df['p_value'])
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    #------------------------------------------
    #           Random Forest 
    #------------------------------------------
    # list contains dictionarys for each fold
    train_mse_records = []
    train_r2_records = []
    val_mse_records = []
    val_r2_records = []
    feature_importance_records = []
    y_val_list = []
    y_val_pred_list = []
    importance_list = []

    # load pre-selected hyper parameters
    best_param_file = open(best_param_dir, 'rb')
    best_param = pickle.load(best_param_file)
    print('hyper-parameters selected from randomized search:')
    print(best_param)

    column_names_X = column_names[0:-1]
    column_names_y = []
    column_names_y.append(column_names[-1]) 

    # run the model in a cross-validation manner
    for train_index, val_index in kf.split(X):
        # load the hyper-parameters into the random forest model
        rf = RandomForestRegressor(#n_estimators=best_param['n_estimators'],
                                    n_estimators=1000,
                                    criterion=best_param['criterion'],
                                    #max_features=best_param['max_features'],
                                    max_features='sqrt',
                                    #max_depth=best_param['max_depth'],
                                    #max_depth = 15,
                                    #min_samples_split=best_param['min_samples_split'],
                                    min_samples_split=4,
                                    #min_samples_leaf=best_param['min_samples_leaf'], 
                                    min_samples_leaf = 4,
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

        # validation labels and predictions to generate scatter plot
        y_val_list.extend(y_val)
        y_val_pred_list.extend(y_val_pred)

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

        #feature_importance_records.append(rf.feature_importances_)
        df_X_val = np_array_to_df(X_val, column_names_X)
        df_y_val = np_array_to_df(np.expand_dims(y_val, axis=1), column_names_y) # add one dimension to y_val
        importance = importances(rf, df_X_val, df_y_val, n_samples=-1, features=grouping_list)
        importance_list.append(importance)
        print('--------------------------------------------------')

    #------------------------------------------
    #           Post-processing 
    #------------------------------------------
    # averaged validation metrics over folds
    print('averaged training mse:', mean(train_mse_records))
    print('averaged training r2 score:', mean(train_r2_records))
    print('averaged validation mse:', mean(val_mse_records))
    print('averaged validation r2 score:', mean(val_r2_records))

    # process the permutation feature ranking data
    plot_features, plot_importances, plot_std = avg_importance(importance_list)

    # bar plot of averaged importances
    y_pos = np.arange(len(plot_features))
    fig, ax = plt.subplots()
    if show_std == 'True':
        ax.barh(y_pos, plot_importances, xerr=plot_std, align='center', color='lightseagreen')
    elif show_std == 'False':
        ax.barh(y_pos, plot_importances, align='center', color='lightseagreen')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_features)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Importance')
    ax.set_title('Permutation feature importances')
    fig.savefig('./figures_permutation_importance/regression_' + dataset + '_use_structual_features_' + use_structual_features + '_grouping_' + grouping + '_showstd_' + show_std + '.png', bbox_inches='tight')
    plt.close('all')

    # scatter plot of the predicted p-values and their true values
    p_x, p_y = scatter_gen(y_val_list, y_val_pred_list)
    plt.scatter(p_x, p_y, c="g", alpha=0.5, marker='.')
    plt.xlabel("True p-values")
    plt.ylabel("Predicted p-values")
    plt.savefig('./figures_scatter/regression_' + dataset + '_use_structual_features_' + use_structual_features + '_grouping_' + grouping + '_showstd_' + show_std + '.png', bbox_inches='tight')
    plt.show()