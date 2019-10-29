'''
Perform 5-fold cross validation and show the classification performance for
the patient dataset.
'''
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

def get_args():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-data_dir',
                        default='../../data/patient_output/patients_latent_truncated_svd.csv',
                        required=False,
                        help='directory of SNP dataframe')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    data_dir = args.data_dir
    df = pd.read_csv(data_dir)

    # report class ratio
    df_pos = pd.concat([df[df['class'] == 1], df[df['class'] == 2], df[df['class'] == 3]])
    df_neg = df[df['class'] == 0]
    num_pos = df_pos.shape[0]
    num_neg = df_neg.shape[0]
    print('number of positive samples:', num_pos)
    print('number of negative samples:', num_neg)

    # data and label
    X = np.array(df.drop(columns = ['class']))
    y = np.array(df['class'])
    y[np.where(y==2)] = 1 # you are 1 if you are sick
    y[np.where(y==3)] = 1 # you are 1 if you are sick
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
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

    # run the model in a cross-validation manner
    for train_index, val_index in skf.split(X, y):
        # load the hyper-parameters into the random forest model
        rf = RandomForestClassifier(n_estimators=1000,
                                    min_samples_split = 0.05,
                                    min_samples_leaf = 0.05,
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
    result_dir = './results/roc_patient_classification' + '.pickle'
    pickle_out = open(result_dir,"wb")
    pickle.dump(roc_records, pickle_out)

