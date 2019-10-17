
#!/bin/bash
#PBS -l nodes=1:ppn=48
#PBS -l walltime=72:00:00
#PBS -q bigmem
#PBS -N rf_snps_cla
#PBS -A loni_omics01
#PBS -j oe

source activate pytorch
cd /work/wshi6/deeplearning-data/gwas-breast-cancer-prj/gwas-breast-cancer/p_value_classification/
python p_value_random_forest.py -dataset erneg -use_structual_features True > ./log/rf_erneg_structual_features_true.log 2>&1
python p_value_random_forest.py -dataset erneg -use_structual_features False > ./log/rf_erneg_structual_features_false.log 2>&1
python p_value_random_forest.py -dataset erpos -use_structual_features True > ./log/rf_erpos_structual_features_true.log 2>&1
python p_value_random_forest.py -dataset erpos -use_structual_features False > ./log/rf_erpos_structual_features_false.log 2>&1
#python p_value_random_forest.py -dataset nih_nodup -use_structual_features True > ./log/rf_nih_nodup_structual_features_true.log 2>&1
#python p_value_random_forest.py -dataset nih_nodup -use_structual_features False > ./log/rf_nih_nodup_structual_features_false.log 2>&1











































