"""
This code is used to find the optimal random forest hyper-parameters for each dataset.
Basically, it does a grid search on the specified hyper-parameters.
"""
import argparse
import numpy as np  
import pandas as pd  
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from time import time
import pickle

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

    parser.add_argument('-num_search_iter',
                        default = 500,
                        required = False)

    parser.add_argument('-scoring_metric',
                        default = None,
                        required = False,
                        help = 'Scoring metric for model selection. See scikit-learn docs.')

    parser.add_argument('-seed',
                        default = 42,
                        required = False)

    return parser.parse_args()

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

if __name__ == "__main__":
    args = get_args()
    dataset = args.dataset
    use_gene_expression = args.use_gene_expression
    num_search_iter =args.num_search_iter
    scoring_metric = args.scoring_metric
    seed = args.seed
    print('dataset:', dataset)
    print('scoring metric for model selection:', scoring_metric)
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

    #-----------------------------------------------------------
    #   Random forest with randomized hyper-paramter searching
    #-----------------------------------------------------------
     # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 500, stop = 2500, num = 10)]
    # which metric to measure the quality of split
    criterions = ['mse', 'mae']
    # Number of features to consider at every split
    max_features = ['sqrt', 'log2']
    max_features.append(None)
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 150, num = 10)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 4, 6, 8, 10, 12]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 6, 8]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    param_grid = {'n_estimators': n_estimators,
                  'criterion': criterions,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'bootstrap': bootstrap,
                  'n_jobs': [-1],
                  'random_state': [seed]}   
    print('hyper paramter search space:')
    print(param_grid)

    # intance of random forest classifier
    clf = RandomForestRegressor()

    # run randomized search
    random_search = RandomizedSearchCV(clf, param_distributions=param_grid,
                                            scoring = scoring_metric,
                                            n_iter=num_search_iter,
                                            cv=5, 
                                            iid=False,
                                            random_state=seed,
                                            verbose=0)

    print('total number of iterations for randomized search: ',num_search_iter)
    start = time()
    random_search.fit(X, y)
    print('Elapsed time for randomized search: ', (time()-start))

    print('results of randomized search:')
    report(random_search.cv_results_)  

    print('best hyper-parameter combinaiton:')
    print(random_search.best_params_)
    pickle_out = open(best_param_dir,"wb")
    pickle.dump(random_search.best_params_, pickle_out)



    
