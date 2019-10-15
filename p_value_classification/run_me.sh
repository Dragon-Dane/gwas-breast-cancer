
#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=72:00:00
#PBS -q workq
#PBS -N rf_snps_cla
#PBS -A loni_omics01
#PBS -j oe

source activate pytorch
cd /work/wshi6/deeplearning-data/gwas-breast-cancer-prj/gwas-breast-cancer/p_value_classification/
python p_value_random_forest_tuning.py -dataset nih -scoring_metric f1 > ./log/rf_tuning_nih.log 2>&1
python p_value_random_forest_tuning.py -dataset erneg -scoring_metric f1 > ./log/rf_tuning_michailidu_erneg.log 2>&1
python p_value_random_forest_tuning.py -dataset erpos -scoring_metric f1 > ./log/rf_tuning_michailidu_erpos.log 2>&1
python p_value_random_forest_tuning.py -dataset nih_nodup -scoring_metric f1 > ./log/rf_tuning_nih_nodup.log 2>&1
python p_value_random_forest.py -dataset nih -use_gene_expression True > ./log/rf_nih_gene_exp_true.log 2>&1
python p_value_random_forest.py -dataset nih -use_gene_expression False > ./log/rf_nih_gene_exp_false.log 2>&1
python p_value_random_forest.py -dataset erneg -use_gene_expression True > ./log/rf_erneg_gene_exp_true.log 2>&1
python p_value_random_forest.py -dataset erneg -use_gene_expression False > ./log/rf_erneg_gene_exp_false.log 2>&1
python p_value_random_forest.py -dataset erpos -use_gene_expression True > ./log/rf_erpos_gene_exp_true.log 2>&1
python p_value_random_forest.py -dataset erpos -use_gene_expression False > ./log/rf_erpos_gene_exp_false.log 2>&1
python p_value_random_forest.py -dataset nih_nodup -use_gene_expression True > ./log/rf_nih_nodup_gene_exp_true.log 2>&1
python p_value_random_forest.py -dataset nih_nodup -use_gene_expression False > ./log/rf_nih_nodup_gene_exp_false.log 2>&1











































