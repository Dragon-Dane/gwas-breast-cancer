import pandas as pd
import numpy as np
import re
import time
import argparse

def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-input',required=True,help='SNPs input file')
    parser.add_argument('-out',required=True,help='output of disorder')
    return parser.parse_args()
    
def disorder(filename,output):
    SNP_dataset = pd.read_csv(filename, delim_whitespace=True, names=['PROTEIN_ID', 'SNP_ID', 'Position', 'wild_type', 'Mutant', 'wt-codon', 'mu-codon'], dtype=object)
    myfile = open(output, 'w')
    for protein, position, wildtype, mutant, snp, wt, mu in zip(SNP_dataset['PROTEIN_ID'], SNP_dataset['Position'], SNP_dataset['wild_type'], SNP_dataset['Mutant'], SNP_dataset['SNP_ID'], SNP_dataset['wt-codon'], SNP_dataset['mu-codon']):
        with open('%s.diso' %(protein), 'r') as f:
            for line in f:
                a7a = list(line.split())
                if str(position) == a7a[0]:
                    myfile.write(protein + '\t' + snp + '\t' + wt + '\t' + mu + '\t' + wildtype + '\t' + mutant + '\t' + str(position) + '\t' + a7a[2] + '\t' + a7a[3] + '\n')
    myfile.close()
if __name__ == "__main__":
    args = getArgs()
    disorder = disorder(args.input,args.out)
    start = time.time()
    end = time.time()
print ('time elapsed:' + str(end - start))
