
#!/bin/bash
#PBS -l nodes=1:ppn=48
#PBS -l walltime=72:00:00
#PBS -q bigmem
#PBS -N rf_snps
#PBS -A loni_omics01
#PBS -j oe

source activate pytorch
cd /work/wshi6/deeplearning-data/gwas-breast-cancer-prj/gwas-breast-cancer/p_value_prediction/
python p_value_random_forest_tuning.py -dataset nih > ./log/rf_tuning_nih.log 2>&1
python p_value_random_forest_tuning.py -dataset erneg > ./log/rf_tuning_michailidu_erneg.log 2>&1
python p_value_random_forest_tuning.py -dataset erpos > ./log/rf_tuning_michailidu_erpos.log 2>&1
python p_value_random_forest.py -dataset nih -use_gene_expression True > ./log/rf_nih_gene_exp_true.log 2>&1
python p_value_random_forest.py -dataset nih -use_gene_expression False > ./log/rf_nih_gene_exp_false.log 2>&1
python p_value_random_forest.py -dataset erneg -use_gene_expression True > ./log/rf_erneg_gene_exp_true.log 2>&1
python p_value_random_forest.py -dataset erneg -use_gene_expression False > ./log/rf_erneg_gene_exp_false.log 2>&1
python p_value_random_forest.py -dataset erpos -use_gene_expression True > ./log/rf_erpos_gene_exp_true.log 2>&1
python p_value_random_forest.py -dataset erpos -use_gene_expression False > ./log/rf_erpos_gene_exp_false.log 2>&1





































