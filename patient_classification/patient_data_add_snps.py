"""
Combine the patient dataset and SNP dataset, then run random forest on the combined patient dataset.
"""

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-snp_dir',
                        default='../../data/original/NIH_part_SNPs_features_nodup.csv',
                        required=False,
                        help='directory of SNP dataframe')

    parser.add_argument('-patient_root_dir',
                        default='../../data/original/NIH_SNPs_features.csv',
                        required=False,
                        help='root directory of the patients')
                    
    return parser.parse_args()

def snp_df_gen(snp_dir):
    """
    Returns a SNP dataframe, the snp ids are used as indexes
    """
    return snp_df

def patient_df_gen(patient_dir), snp_df:
    """
    Returns an falttened dataframe of a single patient. The attributes 
    of the SNPs are added to the original patient dataframe.
    """
    return patient_df

class RandomForestModel:
    def __init__(self):
        self.__train_acc_records = []
        self.__val_acc_records = []
        self.__val_precision_records = []
        self.__val_recall_records = []
        self.__val_mcc_records = []
        self.__roc_records = []
        self.__thresholds_records = []

    #def train(X, y):

    #def report_results():

    #def plot_roc():


if __name__ == "__main__":
    args = get_args()
    snp_dir = args.snp_dir
    patient_root_dir = args.patient_root_dir

    snp_df = snp_df_gen(snp_dir) # dataframe of SNPs indexed by SNP-id

    # list of files of patients
    # Number of patients
    m = 1
    
    # Put all the patient's attributes in a large matirx. Could be very large, use bigmem nodes. 
    for i in range(m):

    # Labels of patients
    
    # Random forest
