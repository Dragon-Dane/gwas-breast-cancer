# Feature engineering
Feature engineering for machine learning

## Gene expression sources
* ERpos to Michailidu ERPos
* ERpos to NIH
* ERneg to Michailidu ERneg

## Codes
* ```insert_gene_exp.py```: read the gene expression information from graph, then add the corresponding gene expression to the csv files according to SNP id.

* ```insert_gene_exp_nih_limeng.py```: This code does two things:
    * Read the gene expression information from graph, then add the corresponding gene expression to the csv files according to SNP id.
    * Read 'diseases_score', 'disgenet_score', and 'malacards_score' from the original nih dataset, and merge with the new NIH dataset where there are no duplicates.

