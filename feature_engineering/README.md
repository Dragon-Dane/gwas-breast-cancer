# Feature engineering
Feature engineering for machine learning

## Gene expression sources
* ERpos to Michailidu ERPos
* ERpos to NIH
* ERneg to Michailidu ERneg

## Codes
* ```insert_gene_exp.py```: read the gene expression information from graph, then add the corresponding gene expression to the csv files according to SNP id.

* ```combine_features.py```: combine the one-hot encoded and binary encoded features into one feature.   
    1. One-hot encoded, combine 'core', 'nis', and 'interface' into 'struct_pos'.
    2. Binary encoded, combine 'hbo','sbr', 'aromatic', and 'hydrophobic' into 'sec_struct'.
    3. One-hot encoded, combine 'coil', 'helix', and 'sheet' into 'interactions'. 

