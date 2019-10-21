
#!/bin/bash
#PBS -l nodes=1:ppn=48
#PBS -l walltime=72:00:00
#PBS -q bigmem
#PBS -N rf_snps_cla
#PBS -A loni_omics01
#PBS -j oe

#source activate pytorch
#cd /work/wshi6/deeplearning-data/gwas-breast-cancer-prj/gwas-breast-cancer/p_value_classification/
python p_value_random_forest_pimp.py -dataset erneg -grouping all_individual > ./log/pimp_rf_erneg_all_individual.log 2>&1
python p_value_random_forest_pimp.py -dataset erneg -grouping all_3_groups > ./log/pimp_rf_erneg_all_3_groups.log 2>&1
python p_value_random_forest_pimp.py -dataset erneg -grouping all_5_groups > ./log/pimp_rf_erneg_all_5_groups.log 2>&1
python p_value_random_forest_pimp.py -dataset erpos -grouping all_individual > ./log/pimp_rf_erpos_all_individual.log 2>&1
python p_value_random_forest_pimp.py -dataset erpos -grouping all_3_groups > ./log/pimp_rf_erpos_all_3_groups.log 2>&1
python p_value_random_forest_pimp.py -dataset erpos -grouping all_5_groups > ./log/pimp_rf_erpos_all_5_groups.log 2>&1


















































