
#!/bin/bash
#PBS -l nodes=1:ppn=48
#PBS -l walltime=72:00:00
#PBS -q bigmem
#PBS -N patient_data_tsne
#PBS -A loni_omics01
#PBS -j oe

source activate pytorch
cd /work/wshi6/deeplearning-data/gwas-breast-cancer-prj/gwas-breast-cancer/feature_engineering/
python patient_t_sne.py > ./log/patient_t_sne.log 2>&1















































