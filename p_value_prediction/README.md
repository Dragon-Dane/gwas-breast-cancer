# Developing notes
Machine learning on the SNPs datasets. Those two datasets are NIH and michailidu genenames disease association. Random forest is used as the classification model.

## Hyper-parameter tuning
The code ```p_value_random_forest_tuning``` conducts randomized searching to find optimal combination of hyper-parameters of random forest. The selected hyper-parameters are stored in ```./best_param/``` folder as pickles. 

## Model inferencing
The code ```p_value_random_forest``` load the selected hyper-parameters and perform 5-fold cross validation. 

## Usage
```./run_me.sh```
