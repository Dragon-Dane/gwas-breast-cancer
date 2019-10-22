
#!/bin/bash
#PBS -l nodes=1:ppn=48
#PBS -l walltime=72:00:00
#PBS -q bigmem
#PBS -N rf_snps_reg
#PBS -A loni_omics01
#PBS -j oe

source activate pytorch
cd /work/wshi6/deeplearning-data/gwas-breast-cancer-prj/gwas-breast-cancer/p_value_regression/
python p_value_random_forest_regression_tuning.py -dataset erneg > ./log/rf_tuning_erneg.log 2>&1
python p_value_random_forest_regression_tuning.py -dataset erpos > ./log/rf_tuning_erpos.log 2>&1













































