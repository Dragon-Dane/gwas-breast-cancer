'''
Using resgression models to predict p-values according to the features of SNPs.
'''
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import roc_curve
from statistics import mean 

#------------------------------------------
#           Data pre-processing
#------------------------------------------
p_value_threshold = 0.005
print('threshold for p-value is set to ', p_value_threshold)
df = pd.read_csv("../data/NIH.csv", delim_whitespace=True) # read data 

df = df[['wildtype_value', 
         'mutant_value', 
         'confidence', 
         'core', 
         'NIS', 
         'Interface', 
         'HBO', 
         'SBR', 
         'Aromatic', 
         'Hydrophobic',  
         'Blosum62_mu', 
         'Blosum62_wt', 
         'helix', 
         'coil', 
         'sheet', 
         'Entropy', 
         'diseases_score', 
         'disgenet_score', 
         'malacards_score', 
         'P-value', 
         'Gene_expression']] #selecting needed columns
#print(df)

# replace 0 in Gene_expression with np.nan because it represents missing value
gene_expressions = df['Gene_expression']
gene_expressions = [np.nan if g==0 else g for g in gene_expressions]
df['Gene_expression'] = gene_expressions

# add a column indicating the prediction result (0 or 1)
p_values = df['P-value']
p_values = np.exp([p_value * (-1) for p_value in p_values]) # convert to original values from -log values  
p_values_binary = [int(p_value <= p_value_threshold) for p_value in p_values]
df = df.drop(columns = ['P-value'])
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
val_acc_records = []
val_precision_records = []
val_recall_records = []
val_f1_records = []
val_mcc_records = []
fpr_records = []
tpr_records = []
thresholds_records = []

# run the model in a cross-validation manner
print('--------------------------------------------------')
for train_index, val_index in skf.split(X, y):
    print('Starting cross-validation on XGBoost model...')
    xgb_clf = xgb.XGBClassifier(n_estimators=3000, verbosity=1, n_jobs=1, scale_pos_weight=scale_pos_weight, random_state=123)
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

    print('--------------------------------------------------')

#------------------------------------------
#           Post-processing 
#------------------------------------------
# averaged validation metrics over folds
print('averaged training accuracy:', mean(train_acc_records))
print('averaged validation accuracy:', mean(val_acc_records))
print('averaged validation precision:', mean(val_precision_records))
print('averaged validation recall:', mean(val_recall_records))
print('averaged validation F1:', mean(val_f1_records))
print('averaged validation MCC:', mean(val_mcc_records))

# rank the features