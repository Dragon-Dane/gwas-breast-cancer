# Developing notes
Machine learning on the SNPs datasets. Those two datasets are NIH and michailidu genenames disease association (MGDA) respectively. Two tree-based methods are used: XGBoost and random forest.

## XGBoost on NIH
1. "Gene_expression" is a ordinal feature (1, 2, 3) with missing values (np.nan).
2. Adding the feature "Gene_expression" actually makes the performance of model worse.
3. The results when the p-value threshold is 0.005 is better than when it is 0.05.

## Random forest on NIH
1. Scikit-learn implementation can not handle missing data well, so the feature "Gene_expression" needs to be removed.

## XGBoost on MGDA
1. ```final_michailidu_genenames_disease_association``` does not contain "Gene_expression".

## Random forest on MGDA