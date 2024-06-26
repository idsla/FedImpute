import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PowerTransformer
from dython.nominal import correlation_ratio
from loguru import logger
from sklearn.datasets import fetch_openml
import numpy as np
from .data_prep_utils import (
	normalization, move_target_to_end, convert_gaussian, drop_unique_cols, one_hot_categorical,
)
from .data_prep_his import (
	process_heart, process_codrna,  process_codon, process_mimiciii_mortality, process_genetic,
	process_mimiciii_mo2, process_mimic_icd, process_mimic_icd2, process_mimic_mo, process_mimic_los
)

########################################################################################################################
# factory function to load dataset
########################################################################################################################
def load_data(dataset_name, normalize=True, verbose=False, threshold=None):

	########################################################################################################################
	# Classification
	########################################################################################################################
	#######################################################################################################################
	# Healthcare Dataset
	#######################################################################################################################
	
	if dataset_name == 'heart':
		return process_heart(pca=True, sample=False)
	elif dataset_name == 'heart_balanced':
		return process_heart(pca=True, sample=True)
	elif dataset_name == 'codrna':
		return process_codrna(normalize, verbose, threshold, sample=False)
	elif dataset_name == 'codrna_balanced':
		return process_codrna(normalize, verbose, threshold, sample=True)
	elif dataset_name == 'codon':
		return process_codon(verbose, threshold)
	elif dataset_name == 'mimiciii_mo':
		return process_mimiciii_mortality()
	elif dataset_name == 'mimiciii_icd':
		return process_mimic_icd2()
	elif dataset_name == 'mimiciii_mo2':
		return process_mimic_mo()
	elif dataset_name == 'mimiciii_los':
		return process_mimic_los()
	elif dataset_name == 'genetic':
		return process_genetic(sample=False)
	elif dataset_name == 'genetic_balanced':
		return process_genetic(sample=True)
	else:
		raise Exception("Unknown dataset name {}".format(dataset_name))
