"""
Random forest for p-value prediction. The feature ranking is done by permutation importance.
"""
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix
from statistics import mean 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from rfpimp import *

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

    parser.add_argument('-p_value_th',
                        default = 0.05,
                        required = False)

    parser.add_argument('-show_std',
                         default = 'False',
                         required = False,
                         choices = ['True', 'False'])

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
    p_value_threshold = float(args.p_value_th)
    show_std = args.show_std
    print('dataset:', dataset)
    print('threshold for p-value is set to ', p_value_threshold)
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
    # add a column indicating the prediction result (0 or 1)
    p_values = df['p_value']
    p_values = np.exp([p_value * (-1) for p_value in p_values]) # convert to original values from -log values  
    p_values_binary = [int(p_value <= p_value_threshold) for p_value in p_values]
    df = df.drop(columns = ['p_value'])
    df['classification_result'] = p_values_binary 

    #calculate class weight
    df_pos = df[df['classification_result'] == 1]
    df_neg = df[df['classification_result'] == 0]
    num_pos = df_pos.shape[0]
    num_neg = df_neg.shape[0]
    print('number of positive samples:', num_pos)
    print('number of negative samples:', num_neg)

    # data and label
    X = np.array(df.drop(columns = ['classification_result']))
    y = np.array(df['classification_result'])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    
    #------------------------------------------
    #           Random Forest 
    #------------------------------------------
    # list contains dictionarys for each fold
    train_acc_records = []
    val_acc_records = []
    val_precision_records = []
    val_recall_records = []
    val_mcc_records = []
    roc_records = []
    thresholds_records = []
    feature_importance_records = []
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
                                    class_weight='balanced',
                                    n_jobs = -1)

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

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
        roc_records.append({'fpr':fpr, 'tpr':tpr, 'auc': auc(fpr, tpr)})

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
    print('averaged training accuracy:', mean(train_acc_records))
    print('averaged validation accuracy:', mean(val_acc_records))
    print('averaged validation precision:', mean(val_precision_records))
    print('averaged validation recall:', mean(val_recall_records))
    print('averaged validation MCC:', mean(val_mcc_records))

    # pickle the ROC info for plotting
    result_dir = './results/roc_' + dataset + '_use_structual_features_' + use_structual_features + '_p_value_' + str(p_value_threshold) + '_grouping_' + grouping + '_showstd_' + show_std +'.pickle'
    pickle_out = open(result_dir,"wb")
    pickle.dump(roc_records, pickle_out)

    # process the permutation feature ranking data
    plot_features, plot_importances, plot_std = avg_importance(importance_list)

    # bar plot of averaged importances
    y_pos = np.arange(len(plot_features))
    #plt.figure(figsize=(16,8))
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
  
    #plt.show()
    fig.savefig('./figures_permutation_performance/random_forest_' + dataset + '_use_structual_features_' + use_structual_features + '_p_value_' + str(p_value_threshold) + '_grouping_' + grouping + '_showstd_' + show_std + '.png', bbox_inches='tight')
    