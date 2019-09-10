python random_forest.py -dataset ERneg -use_gene_expression True > ./log/p_value_random_forest_ERneg_with_gene_expression.log
python random_forest.py -dataset ERneg -use_gene_expression False > ./log/p_value_random_forest_ERneg_without_gene_expression.log
python random_forest.py -dataset ERpos -use_gene_expression True > ./log/p_value_random_forest_ERpos_with_gene_expression.log
python random_forest.py -dataset ERpos -use_gene_expression False > ./log/p_value_random_forest_ERpos_without_gene_expression.log



