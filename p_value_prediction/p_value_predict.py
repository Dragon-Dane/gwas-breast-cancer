'''
Using resgression models to predict p-values according to the features of SNPs.
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix

#------------------------------------------
#           Data pre-processing
#------------------------------------------
p_value_threshold = 0.05
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
print(df)

# add a column indicating the prediction result (0 or 1)
p_values = df['P-value']
p_values_binary = [int(p_value <= 0.05) for p_value in p_values]
df = df.drop(columns = ['P-value'])
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
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

#------------------------------------------
#           Random Forest 
#------------------------------------------
rf = RandomForestClassifier(n_estimators=1000,
                            max_depth=None,
                            min_samples_split=100,
                            random_state=0,
                            max_features = "auto",
                            criterion = "gini",
                            class_weight = class_weight,
                            n_jobs = -1)

# list contains dictionarys for each fold
val_acc_records = []
val_precision_records = []
val_recall_records = []
val_mcc_records = []
fpr_records = []
tpr_records = []
thresholds_records = []

# run the model in a cross-validation manner
for train_index, val_index in skf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    rf.fit(X_train, y_train)# train the random forest
    
    # train the rf on the training data
    print('training the random forest...')
    rf.fit(X_train, y_train)
    print('training finished.')

    # training performance
    train_acc = rf.score(X_train, y_train)
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

#------------------------------------------
#           Gradient Boosting 
#------------------------------------------


#------------------------------------------
#           Neural Network
#------------------------------------------
#print('Training a neural network...')

# convert predicted p-values to classification results
# metrics for training
# metrics for vlication