import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, PowerTransformer, LabelEncoder, StandardScaler, RobustScaler
)
from dython.nominal import correlation_ratio
from loguru import logger
from sklearn.datasets import fetch_openml
import numpy as np
from .data_prep_utils import (
	convert_gaussian, normalization, drop_unique_cols, one_hot_categorical, move_target_to_end,
)
from sklearn.feature_selection import mutual_info_classif
from imblearn.under_sampling import RandomUnderSampler


def process_heart(verbose=False, threshold=None, pca=False, sample = False):

	if threshold is None:
		threshold = 0.1
	df = pd.read_csv('./data/heart/heart_2020_cleaned.csv')
	target_col = 'HeartDisease'
	df['HeartDisease'] = df['HeartDisease'].map({"No": 0, "Yes": 1})
	df[target_col].value_counts()
	df = move_target_to_end(df, target_col)

	categorical_columns = [
		'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 
		'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 
		'Asthma', 'KidneyDisease', 'SkinCancer'
	]

	numerical_cols = [col for col in df.columns if col not in categorical_columns and col != target_col]
	print(numerical_cols)
	oh_encoder = OneHotEncoder(max_categories=10, drop='first', sparse_output = False)
	X_cat = oh_encoder.fit_transform(df[categorical_columns])
	X_num = df[numerical_cols].values
	robust_scaler = MinMaxScaler()
	X_num = robust_scaler.fit_transform(df[numerical_cols])
	X = np.concatenate([X_num, X_cat], axis=1)

	# pca
	pca = PCA(n_components=0.95)
	X_new = pca.fit_transform(X)
	y_new = df[target_col].values

	# new dataframe
	df_new = pd.DataFrame(np.concatenate([X_new, y_new.reshape(-1, 1)], axis=1))
	df_new.columns = df_new.columns.astype(str)

	target_col = df_new.columns[-1]
	df_new = move_target_to_end(df_new, target_col)
	df_new[target_col] = pd.factorize(df_new[target_col])[0]
	df_new = convert_gaussian(df_new, target_col)
	df_new = normalization(df_new, target_col)
	print(df_new.shape)

	data = df_new

	if sample:
		# sampling
		target_col = data.columns[-1]
		print(data.shape)
		under = RandomUnderSampler(random_state=42)
		X = data.drop([target_col], axis=1).values
		X_new, y_new = under.fit_resample(X, data[target_col].values)
		data = pd.DataFrame(np.concatenate([X_new, y_new.reshape(-1, 1)], axis = 1))
		print(data.shape)
		target_col = data.columns[-1]
		df0 = data[data[target_col] == 0]
		df1 = data[data[target_col] == 1]
		if df0.shape[0] > df1.shape[0]:
			df0 = df0.sample(n=df1.shape[0], random_state=42)
		elif df0.shape[0] < df1.shape[0]:
			df1 = df1.sample(n=df0.shape[0], random_state=42)
		
		data = pd.concat([df0, df1], axis=0)
		print(data.shape)
	
	if len(data) >= 20000:
		data = data.sample(n = 20000, random_state=42).reset_index(drop=True)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	print(data.shape)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


def process_codrna(normalize=True, verbose=False, threshold=None, sample=True, gaussian=True):

	if threshold is None:
		threshold = 0.1

	data = pd.read_csv('./data/codrna/codrna.csv')

	target_col = 'Y'
	data[target_col] = pd.factorize(data[target_col])[0]
	data = data.dropna()

	if gaussian:
		data = convert_gaussian(data, target_col)

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	if sample:
		data_y0 = data[data[target_col] == 0]
		data_y1 = data[data[target_col] == 1]
		if data_y0.shape[0] > data_y1.shape[0]:
			data_y0 = data_y0.sample(n=data_y1.shape[0], random_state=0)
		elif data_y0.shape[0] < data_y1.shape[0]:
			data_y1 = data_y1.sample(n=data_y0.shape[0], random_state=0)
		data = pd.concat([data_y0, data_y1], axis=0).reset_index(drop=True)
	
	if len(data) >= 20000:
		data = data.sample(n=20000).reset_index(drop=True)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


def process_codon(verbose=False, threshold=None):
	if threshold is None:
		threshold = 0.1
	data = pd.read_csv("./data/codon/codon_usage.csv", sep=',', low_memory=False)

	data = data.dropna()
	data = data.replace("non-B hepatitis virus", 0)
	data = data.replace("12;I", 0)
	data = data.replace('-', 0)
	#data.columns = [str(i) for i in range(data.sh
	data = data.drop(['SpeciesID', 'Ncodons', 'SpeciesName', 'DNAtype'], axis=1)
	target_col = 'Kingdom'
	data = data[data[target_col] != 'plm']
	data[target_col], codes = pd.factorize(data[target_col])
	data = move_target_to_end(data, target_col)
	data = normalization(data, target_col)

	# pca
	pca = PCA(n_components=0.9)
	X = pca.fit_transform(data.drop(target_col, axis=1).values)
	y = data[target_col].values

	# new dataframe
	df_new = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], axis=1))
	df_new.columns = df_new.columns.astype(str)

	target_col = df_new.columns[-1]
	data = df_new
	print(data.shape)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config



def process_mimiciii_mortality():
	
	# read data
	df_patient = pd.read_csv('data/mimiciii/PATIENTS.csv').set_index('ROW_ID')
	df_admissions = pd.read_csv('data/mimiciii/ADMISSIONS.csv').set_index('ROW_ID')
	df_prescriptions = pd.read_csv('data/mimiciii/PRESCRIPTIONS.csv', low_memory = False).set_index('ROW_ID')
	df_prescriptions['PAID'] = df_prescriptions['SUBJECT_ID'].astype(str) + '_' + df_prescriptions['HADM_ID'].astype(str)
	df_prescriptions['STARTDATE'] = pd.to_datetime(df_prescriptions['STARTDATE'])
	df_prescriptions['ENDDATE'] = pd.to_datetime(df_prescriptions['ENDDATE'])
	print(df_prescriptions.shape)

	# patient admission table
	df_patient_admission = df_patient.merge(df_admissions, on='SUBJECT_ID', how='inner')
	df_patient_admission = df_patient_admission.drop([
		'DOD_HOSP', 'DOD_SSN', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG'], axis=1)
	df_patient_admission['PAID'] = df_patient_admission['SUBJECT_ID'].astype(str) + '_' + df_patient_admission['HADM_ID'].astype(str)
	df_patient_admission['ADMITTIME'] = pd.to_datetime(df_patient_admission['ADMITTIME'])
	df_patient_admission['DISCHTIME'] = pd.to_datetime(df_patient_admission['DISCHTIME'])
	df_patient_admission['EDREGTIME'] = pd.to_datetime(df_patient_admission['EDREGTIME'])
	df_patient_admission['EDOUTTIME'] = pd.to_datetime(df_patient_admission['EDOUTTIME'])
	df_patient_admission['DOB'] = pd.to_datetime(df_patient_admission['DOB'])
	df_patient_admission['DOD'] = pd.to_datetime(df_patient_admission['DOD'])
	admission_count = df_patient_admission.groupby('SUBJECT_ID')['HADM_ID'].count()
	df_patient_admission['ADMISSION_COUNT'] = df_patient_admission['SUBJECT_ID'].map(admission_count)
	print(df_patient_admission.shape)

	# admission time
	def update_admission_time(row):
		if pd.isna(row['EDREGTIME']):
			return row['ADMITTIME']
		else:
			if (row['ADMITTIME'] - row['EDREGTIME']) > np.timedelta64(0, 's'):
				return row['EDREGTIME']
			else:
				return row['ADMITTIME']

	def update_disch_time(row):
		if pd.isna(row['EDOUTTIME']):
			return row['DISCHTIME']
		else:
			if (row['DISCHTIME'] - row['EDOUTTIME']) > np.timedelta64(0, 's'):
				return row['DISCHTIME']
			else:
				return row['EDOUTTIME']

	df_patient_admission['ADMITTIME_NEW'] = df_patient_admission.apply(lambda x: update_admission_time(x), axis=1)
	df_patient_admission['DISCHTIME_NEW'] = df_patient_admission.apply(lambda x: update_disch_time(x), axis=1)
	length_of_stay = ((df_patient_admission['DISCHTIME_NEW'] - df_patient_admission['ADMITTIME_NEW'])/np.timedelta64(1, 's'))
	df_patient_admission['LENGTH_OF_STAY'] = length_of_stay
	df_patient_admission = df_patient_admission[df_patient_admission['LENGTH_OF_STAY'] > 0]

	# mortality and only consider last time admission
	mortality = (df_patient_admission['DOD'] - df_patient_admission['DISCHTIME_NEW'])/np.timedelta64(1, 'D')
	df_patient_admission['mortality'] = mortality < 365*2
	df_patient_admission = df_patient_admission.loc[df_patient_admission.groupby('SUBJECT_ID')['ADMITTIME_NEW'].idxmax()].copy()
	target_col = 'mortality'

	# feature engineering
	age = (df_patient_admission['ADMITTIME_NEW'].dt.year - df_patient_admission['DOB'].dt.year)
	age[age >= 90] = 90

	df_patient_admission['AGE'] = age
	df_patient_admission['AGE_IND'] = df_patient_admission['AGE'].apply(lambda x: 1 if x >= 65 else 0)
	df_patient_admission['GENDER'] = df_patient_admission['GENDER'].apply(lambda x: 1 if x == 'M' else 0)

	df_patient_admission.fillna('None', inplace=True)

	cat_cols = [
		'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS',
		'ETHNICITY'
	]

	drop_cols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'EDREGTIME', 'EDOUTTIME', 'DIAGNOSIS', 
	      'DOB', 'DOD', 'EXPIRE_FLAG']

	df_patient_admission.drop(columns=drop_cols, inplace=True)
	print(df_patient_admission.shape)

	# patient information pca
	df_cat = df_patient_admission[cat_cols]
	oh = OneHotEncoder(sparse_output=False, drop="if_binary", max_categories=15)
	oh.fit(df_cat)
	df_cat = oh.transform(df_cat)
	df_cat = pd.DataFrame(df_cat, columns = oh.get_feature_names_out())
	df_patient_admission = pd.concat([df_patient_admission.drop(cat_cols, axis=1).reset_index(drop = True), df_cat], axis=1)
	print(df_patient_admission.shape)

	# fetch drugs 48 hours after admission
	df_pa_date = df_patient_admission[['PAID', 'ADMITTIME_NEW', 'DISCHTIME_NEW']]
	df_prescriptions_join = df_prescriptions.merge(df_pa_date, how='left', on='PAID')
	df_prescriptions_join.dropna(subset=['STARTDATE', 'ENDDATE'], inplace=True)
	time_diff = (df_prescriptions_join['STARTDATE'] - df_prescriptions_join['ADMITTIME_NEW'])/np.timedelta64(1,'D')
	df_prescriptions_join_filtered = df_prescriptions_join[(time_diff <= 2)]

	# get drugs
	drugs = df_prescriptions_join_filtered.groupby('PAID')['DRUG'].apply(lambda x: x.tolist()).reset_index(name='DRUGS')
	drugs_explode = drugs.explode('DRUGS')
	drugs_explode['occurrence'] = 1
	drugs_new = pd.pivot_table(drugs_explode, index='PAID', columns='DRUGS', values = 'occurrence', fill_value=0)
	drugs_new.reset_index(inplace=True)

	# pca drugs
	drugs_pca = drugs_new.drop('PAID', axis = 1)
	pca = PCA(n_components=40)
	pca.fit(drugs_pca)
	drugs_pca = pca.transform(drugs_pca)
	drugs_pca = pd.DataFrame(drugs_pca)
	drugs_new = pd.concat([drugs_new['PAID'], drugs_pca], axis = 1)
	print(drugs_new.shape)

	# join drugs pca with patient admission
	df_result = df_patient_admission.copy()
	df_result = df_result.drop(['PAID', 'ADMITTIME_NEW', 'DISCHTIME_NEW'], axis=1)
	target_col = 'mortality'
	data = df_result
	data = move_target_to_end(data, target_col)

	# scale
	scaler = StandardScaler()
	num_cols = ['LENGTH_OF_STAY', 'AGE']
	data[num_cols] = scaler.fit_transform(data[num_cols])

	# # pca
	pca = PCA(n_components=20)
	pca.fit(data.drop(target_col, axis=1))
	data_pca = pca.transform(data.drop(target_col, axis=1))

	# final table
	data_pca = pd.DataFrame(data_pca)
	data = pd.concat([df_patient_admission['PAID'], data_pca, data[target_col]], axis = 1)
	data_final = data.merge(drugs_new, on = 'PAID', how = 'inner')
	data = data_final.drop('PAID', axis = 1)
	data = move_target_to_end(data, target_col)
	data.columns = [str(col) for col in data.columns]
	print(data.shape)

	data = convert_gaussian(data, target_col)
	data = normalization(data, target_col)

	data = data.sample(n = 20000, random_state = 42)
	data[target_col] = data[target_col].astype(int)
	data.reset_index(drop=True, inplace=True)

	data_config = {
		'target': target_col,
		'important_features_idx': [],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	print(data.shape)
	print(data[target_col].value_counts())
	
	return data, data_config


def process_mimiciii_mo2():
	
	df_patient = pd.read_csv('data/mimiciii/PATIENTS.csv').set_index('ROW_ID')
	df_admissions = pd.read_csv('data/mimiciii/ADMISSIONS.csv').set_index('ROW_ID')

	###########################################################################################################################
	# patient informatin
	df_patient_admission = df_patient.merge(df_admissions, on='SUBJECT_ID', how='inner')
	df_patient_admission = df_patient_admission.drop([
		'DOD_HOSP', 'DOD_SSN', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG'], axis=1)
	df_patient_admission['PAID'] = df_patient_admission['SUBJECT_ID'].astype(str) + '_' + df_patient_admission['HADM_ID'].astype(str)
	df_patient_admission['ADMITTIME'] = pd.to_datetime(df_patient_admission['ADMITTIME'])
	df_patient_admission['DISCHTIME'] = pd.to_datetime(df_patient_admission['DISCHTIME'])
	df_patient_admission['EDREGTIME'] = pd.to_datetime(df_patient_admission['EDREGTIME'])
	df_patient_admission['EDOUTTIME'] = pd.to_datetime(df_patient_admission['EDOUTTIME'])
	df_patient_admission['DOB'] = pd.to_datetime(df_patient_admission['DOB'])
	df_patient_admission['DOD'] = pd.to_datetime(df_patient_admission['DOD'])
	admission_count = df_patient_admission.groupby('SUBJECT_ID')['HADM_ID'].count()
	df_patient_admission['ADMISSION_COUNT'] = df_patient_admission['SUBJECT_ID'].map(admission_count)

	# admission time
	def update_admission_time(row):
		if pd.isna(row['EDREGTIME']):
			return row['ADMITTIME']
		else:
			if (row['ADMITTIME'] - row['EDREGTIME']) > np.timedelta64(0, 's'):
				return row['EDREGTIME']
			else:
				return row['ADMITTIME']

	def update_disch_time(row):
		if pd.isna(row['EDOUTTIME']):
			return row['DISCHTIME']
		else:
			if (row['DISCHTIME'] - row['EDOUTTIME']) > np.timedelta64(0, 's'):
				return row['DISCHTIME']
			else:
				return row['EDOUTTIME']

	df_patient_admission['ADMITTIME_NEW'] = df_patient_admission.apply(lambda x: update_admission_time(x), axis=1)
	df_patient_admission['DISCHTIME_NEW'] = df_patient_admission.apply(lambda x: update_disch_time(x), axis=1)
	length_of_stay = ((df_patient_admission['DISCHTIME_NEW'] - df_patient_admission['ADMITTIME_NEW'])/np.timedelta64(1, 's'))
	df_patient_admission['LENGTH_OF_STAY'] = length_of_stay
	df_patient_admission = df_patient_admission[df_patient_admission['LENGTH_OF_STAY'] > 0]

	# mortality and only consider last time admission
	mortality = (df_patient_admission['DOD'] - df_patient_admission['DISCHTIME_NEW'])/np.timedelta64(1, 'D')
	df_patient_admission['mortality'] = mortality < 365*2
	df_patient_admission = df_patient_admission.loc[df_patient_admission.groupby('SUBJECT_ID')['ADMITTIME_NEW'].idxmax()].copy()

	# feature engineering
	age = (df_patient_admission['ADMITTIME_NEW'].dt.year - df_patient_admission['DOB'].dt.year)
	age[age >= 90] = 90

	df_patient_admission['AGE'] = age
	df_patient_admission['AGE_IND'] = df_patient_admission['AGE'].apply(lambda x: 1 if x >= 65 else 0)
	df_patient_admission['GENDER'] = df_patient_admission['GENDER'].apply(lambda x: 1 if x == 'M' else 0)
	df_patient_admission.fillna('None', inplace=True)

	cat_cols = [
		'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS',
		'ETHNICITY'
	]

	drop_cols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'EDREGTIME', 'EDOUTTIME', 'DIAGNOSIS', 'DOB', 'DOD', 'EXPIRE_FLAG']

	df_patient_admission.drop(columns=drop_cols, inplace=True)
	df_patient_admission.reset_index(drop=True, inplace=True)

	# patient information pca
	df_cat = df_patient_admission[cat_cols]
	oh = OneHotEncoder(sparse_output=False, drop="if_binary", max_categories=15)
	oh.fit(df_cat)
	df_cat = oh.transform(df_cat)
	df_cat = pd.DataFrame(df_cat, columns = oh.get_feature_names_out())

	df_num = df_patient_admission[['LENGTH_OF_STAY', 'AGE']].copy()
	scaler = StandardScaler()
	num_cols = ['LENGTH_OF_STAY', 'AGE']
	df_num[num_cols] = scaler.fit_transform(df_num[num_cols])

	df_patient_info = pd.concat([df_patient_admission['PAID'], df_cat, df_num], axis=1)
	print(df_patient_info.shape)

	patient_info_pca = convert_pca(df_patient_info, 10, prefix = 'pa')

	###############################################################################################################################
	# Drugs
	# fetch drugs 48 hours after admission
	df_pa_date = df_patient_admission[['PAID', 'ADMITTIME_NEW', 'DISCHTIME_NEW']]
	df_prescriptions = pd.read_csv('data/mimiciii/PRESCRIPTIONS.csv', low_memory = False).set_index('ROW_ID')
	df_prescriptions['PAID'] = df_prescriptions['SUBJECT_ID'].astype(str) + '_' + df_prescriptions['HADM_ID'].astype(str)
	df_prescriptions['STARTDATE'] = pd.to_datetime(df_prescriptions['STARTDATE'])
	df_prescriptions['ENDDATE'] = pd.to_datetime(df_prescriptions['ENDDATE'])
	df_prescriptions_join = df_prescriptions.merge(df_pa_date, how='left', on='PAID')
	df_prescriptions_join.dropna(subset=['STARTDATE', 'ENDDATE'], inplace=True)
	time_diff = (df_prescriptions_join['STARTDATE'] - df_prescriptions_join['ADMITTIME_NEW'])/np.timedelta64(1,'D')
	df_prescriptions_join_filtered = df_prescriptions_join[(time_diff <= 2)]

	# get drugs
	drugs_counts = df_prescriptions_join_filtered.groupby(['PAID', 'DRUG']).size().reset_index(name='occurrence')
	drugs = pd.pivot_table(drugs_counts, index='PAID', columns='DRUG', values = 'occurrence', fill_value=0)
	drugs.reset_index(inplace=True)

	# pca
	drugs = drugs.merge(df_patient_admission[['PAID']], how='right', on='PAID')
	drugs.fillna(0, inplace=True)
	print(drugs.shape)

	drugs_pca = convert_pca(drugs, 20, prefix = 'drug')

	###############################################################################################################################
	# Diagnosis
	df_diagnosis = pd.read_csv('data/mimiciii/DIAGNOSES_ICD.csv')
	df_diagnosis['PAID'] = df_diagnosis['SUBJECT_ID'].astype(str) + '_' + df_diagnosis['HADM_ID'].astype(str)
	df_diagnosis = df_diagnosis[['PAID', 'ICD9_CODE']]
	df_diagnosis = df_diagnosis.dropna()
	df_diagnosis['ICD9_GROUP'] = df_diagnosis['ICD9_CODE'].apply(lambda row: icd_group(row))
	df_diagnosis = df_diagnosis.drop('ICD9_CODE', axis=1)

	diagnosis_counts = df_diagnosis.groupby(['PAID', 'ICD9_GROUP']).size().reset_index(name='occurrence')
	diagnosis = pd.pivot_table(diagnosis_counts, index='PAID', columns='ICD9_GROUP', values = 'occurrence', fill_value=0)
	diagnosis.reset_index(inplace=True)
	diagnosis = diagnosis.merge(df_patient_admission[['PAID']], how='right', on='PAID')
	diagnosis.fillna(0, inplace=True)
	print(diagnosis.shape)

	diagnosis_pca = convert_pca(diagnosis, 10, prefix = 'diag')

	###############################################################################################################################
	# Procedure
	df_procedure = pd.read_csv('data/mimiciii/PROCEDUREEVENTS_MV.csv', low_memory = False).set_index('ROW_ID')
	procedure = convert_df(df_procedure)
	procedure = procedure.merge(df_patient_admission[['PAID']], how='right', on='PAID')
	procedure.fillna(0, inplace=True)
	print(procedure.shape)

	procedures_pca = convert_pca(procedure, 10, prefix = 'proc')

	###############################################################################################################################
	# Input events
	df_input = pd.read_csv('data/mimiciii/INPUTEVENTS_MV.csv', low_memory = False).set_index('ROW_ID')
	inputevents = convert_df(df_input)
	inputevents = inputevents.merge(df_patient_admission[['PAID']], how='right', on='PAID')
	inputevents.fillna(0, inplace=True)
	print(inputevents.shape)

	inputevents_pca = convert_pca(inputevents, 10, prefix = 'input')

	###############################################################################################################################
	# join drugs pca with patient admission
	df_result = patient_info_pca.copy()
	df_result = pd.concat([df_result, df_patient_admission['mortality']], axis = 1)
	df_result = df_result.merge(drugs_pca, on = 'PAID', how = 'left')
	df_result = df_result.merge(inputevents_pca, on = 'PAID', how = 'left')
	df_result = df_result.merge(diagnosis_pca, on = 'PAID', how = 'left')
	df_result = df_result.merge(procedures_pca, on = 'PAID', how = 'left')
	df_result = df_result.fillna(0)
	print(df_result.shape)

	target_col = 'mortality'
	data = df_result
	data = data.drop('PAID', axis = 1)
	data.columns = [str(col) for col in data.columns]
	data = move_target_to_end(data, target_col)

	data.columns = [str(col) for col in data.columns]
	data = convert_gaussian(data, target_col)
	data = normalization(data, target_col)
	data = data.sample(n = 20000, random_state = 42)
	data[target_col] = data[target_col].astype(int)
	data.reset_index(drop=True, inplace=True)

	data_config = {
		'target': target_col,
		'important_features_idx': [],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	print(data.shape)
	print(data[target_col].value_counts())
	data.to_csv('data/mimiciii/mimiciii_mo2.csv', index=False)
	
	return data, data_config


def process_mimic_icd():
	df_diagnosis = pd.read_csv('data/mimiciii/DIAGNOSES_ICD.csv')
	df_diagnosis = df_diagnosis[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'SEQ_NUM']]
	df_diagnosis = df_diagnosis.dropna()
	df_diagnosis['PAID'] = df_diagnosis['SUBJECT_ID'].astype(str) + '_' + df_diagnosis['HADM_ID'].astype(str)
	df_diagnosis['ICD9_GROUP'] = df_diagnosis['ICD9_CODE'].apply(lambda row: icd_group(row))
	merged_cats = [16, 17, 11, 12, 13, 20, 13, 15, 4, 10]
	most_frequent_icd = df_diagnosis.groupby(['PAID'])['ICD9_GROUP'].apply(lambda x: x.mode().iloc[0]).reset_index()
	most_frequent_icd['ICD9_GROUP'] = most_frequent_icd['ICD9_GROUP'].apply(lambda x: x if x not in merged_cats else 21)
	most_frequent_icd['ICD9_GROUP'].value_counts()
	print(most_frequent_icd.shape)

	###########################################################################################################################
	# patient informatin
	df_patient = pd.read_csv('data/mimiciii/PATIENTS.csv').set_index('ROW_ID')
	df_admissions = pd.read_csv('data/mimiciii/ADMISSIONS.csv').set_index('ROW_ID')
	df_patient_admission = df_patient.merge(df_admissions, on='SUBJECT_ID', how='inner')
	df_patient_admission = df_patient_admission.drop([
		'DOD_HOSP', 'DOD_SSN', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG'], axis=1)
	df_patient_admission['PAID'] = df_patient_admission['SUBJECT_ID'].astype(str) + '_' + df_patient_admission['HADM_ID'].astype(str)
	df_patient_admission['ADMITTIME'] = pd.to_datetime(df_patient_admission['ADMITTIME'])
	df_patient_admission['DISCHTIME'] = pd.to_datetime(df_patient_admission['DISCHTIME'])
	df_patient_admission['EDREGTIME'] = pd.to_datetime(df_patient_admission['EDREGTIME'])
	df_patient_admission['EDOUTTIME'] = pd.to_datetime(df_patient_admission['EDOUTTIME'])
	df_patient_admission['DOB'] = pd.to_datetime(df_patient_admission['DOB'])
	df_patient_admission['DOD'] = pd.to_datetime(df_patient_admission['DOD'])
	admission_count = df_patient_admission.groupby('SUBJECT_ID')['HADM_ID'].count()
	df_patient_admission['ADMISSION_COUNT'] = df_patient_admission['SUBJECT_ID'].map(admission_count)

	# admission time
	def update_admission_time(row):
		if pd.isna(row['EDREGTIME']):
			return row['ADMITTIME']
		else:
			if (row['ADMITTIME'] - row['EDREGTIME']) > np.timedelta64(0, 's'):
				return row['EDREGTIME']
			else:
				return row['ADMITTIME']

	def update_disch_time(row):
		if pd.isna(row['EDOUTTIME']):
			return row['DISCHTIME']
		else:
			if (row['DISCHTIME'] - row['EDOUTTIME']) > np.timedelta64(0, 's'):
				return row['DISCHTIME']
			else:
				return row['EDOUTTIME']

	df_patient_admission['ADMITTIME_NEW'] = df_patient_admission.apply(lambda x: update_admission_time(x), axis=1)
	df_patient_admission['DISCHTIME_NEW'] = df_patient_admission.apply(lambda x: update_disch_time(x), axis=1)
	length_of_stay = ((df_patient_admission['DISCHTIME_NEW'] - df_patient_admission['ADMITTIME_NEW'])/np.timedelta64(1, 's'))
	df_patient_admission['LENGTH_OF_STAY'] = length_of_stay
	df_patient_admission = df_patient_admission[df_patient_admission['LENGTH_OF_STAY'] > 0]

	# mortality and only consider last time admission
	mortality = (df_patient_admission['DOD'] - df_patient_admission['DISCHTIME_NEW'])/np.timedelta64(1, 'D')
	df_patient_admission['mortality'] = mortality < 365*2
	#df_patient_admission = df_patient_admission.loc[df_patient_admission.groupby('SUBJECT_ID')['ADMITTIME_NEW'].idxmax()].copy()

	# feature engineering
	age = (df_patient_admission['ADMITTIME_NEW'].dt.year - df_patient_admission['DOB'].dt.year)
	age[age >= 90] = 90

	df_patient_admission['AGE'] = age
	df_patient_admission['AGE_IND'] = df_patient_admission['AGE'].apply(lambda x: 1 if x >= 65 else 0)
	df_patient_admission['GENDER'] = df_patient_admission['GENDER'].apply(lambda x: 1 if x == 'M' else 0)
	df_patient_admission.fillna('None', inplace=True)

	cat_cols = [
		'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS',
		'ETHNICITY'
	]

	drop_cols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'EDREGTIME', 'EDOUTTIME', 'DIAGNOSIS', 'DOB', 'DOD', 'EXPIRE_FLAG']

	df_patient_admission.drop(columns=drop_cols, inplace=True)
	df_patient_admission.reset_index(drop=True, inplace=True)

	# patient information pca
	df_cat = df_patient_admission[cat_cols]
	oh = OneHotEncoder(sparse_output=False, drop="if_binary", max_categories=15)
	oh.fit(df_cat)
	df_cat = oh.transform(df_cat)
	df_cat = pd.DataFrame(df_cat, columns = oh.get_feature_names_out())

	df_num = df_patient_admission[['LENGTH_OF_STAY', 'AGE']].copy()
	scaler = StandardScaler()
	num_cols = ['LENGTH_OF_STAY', 'AGE']
	df_num[num_cols] = scaler.fit_transform(df_num[num_cols])

	df_patient_info = pd.concat([df_patient_admission['PAID'], df_cat, df_num], axis=1)
	print(df_patient_info.shape)

	patient_info_pca = convert_pca(df_patient_info, 10, prefix = 'pa')

	###############################################################################################################################
	# Drugs
	# fetch drugs 48 hours after admission
	df_pa_date = df_patient_admission[['PAID', 'ADMITTIME_NEW', 'DISCHTIME_NEW']]
	df_prescriptions = pd.read_csv('data/mimiciii/PRESCRIPTIONS.csv', low_memory = False).set_index('ROW_ID')
	df_prescriptions['PAID'] = df_prescriptions['SUBJECT_ID'].astype(str) + '_' + df_prescriptions['HADM_ID'].astype(str)
	df_prescriptions['STARTDATE'] = pd.to_datetime(df_prescriptions['STARTDATE'])
	df_prescriptions['ENDDATE'] = pd.to_datetime(df_prescriptions['ENDDATE'])
	df_prescriptions_join = df_prescriptions.merge(df_pa_date, how='left', on='PAID')
	df_prescriptions_join.dropna(subset=['STARTDATE', 'ENDDATE'], inplace=True)
	time_diff = (df_prescriptions_join['STARTDATE'] - df_prescriptions_join['ADMITTIME_NEW'])/np.timedelta64(1,'D')
	df_prescriptions_join_filtered = df_prescriptions_join[(time_diff <= 2)]

	# get drugs
	drugs_counts = df_prescriptions_join_filtered.groupby(['PAID', 'DRUG']).size().reset_index(name='occurrence')
	drugs = pd.pivot_table(drugs_counts, index='PAID', columns='DRUG', values = 'occurrence', fill_value=0)
	drugs.reset_index(inplace=True)

	# pca
	drugs = drugs.merge(df_patient_admission[['PAID']], how='right', on='PAID')
	drugs.fillna(0, inplace=True)
	print(drugs.shape)

	drugs_pca = convert_pca(drugs, 10, prefix = 'drug')

	###############################################################################################################################
	# Procedure
	df_procedure = pd.read_csv('data/mimiciii/PROCEDUREEVENTS_MV.csv', low_memory = False).set_index('ROW_ID')
	procedure = convert_df(df_procedure)
	procedure = procedure.merge(df_patient_admission[['PAID']], how='right', on='PAID')
	procedure.fillna(0, inplace=True)
	print(procedure.shape)

	procedures_pca = convert_pca(procedure, 10, prefix = 'proc')

	###############################################################################################################################
	# Input events
	df_input = pd.read_csv('data/mimiciii/INPUTEVENTS_MV.csv', low_memory = False).set_index('ROW_ID')
	inputevents = convert_df(df_input)
	inputevents = inputevents.merge(df_patient_admission[['PAID']], how='right', on='PAID')
	inputevents.fillna(0, inplace=True)
	print(inputevents.shape)

	inputevents_pca = convert_pca(inputevents, 10, prefix = 'input')

	###############################################################################################################################
	# merge all tables
	df_result = most_frequent_icd.copy()
	df_result = df_result.merge(patient_info_pca, on = 'PAID', how = 'inner')
	df_result = df_result.merge(drugs_pca, on = 'PAID', how = 'inner')
	df_result = df_result.merge(inputevents_pca, on = 'PAID', how = 'inner')
	df_result = df_result.merge(procedures_pca, on = 'PAID', how = 'inner')
	print(df_result.shape)

	target_col = 'ICD9_GROUP'
	data = df_result
	data = data.drop('PAID', axis = 1)
	data.columns = [str(col) for col in data.columns]
	data = move_target_to_end(data, target_col)
	data[target_col] = pd.factorize(data[target_col])[0]

	data.columns = [str(col) for col in data.columns]
	data = convert_gaussian(data, target_col)
	data = normalization(data, target_col)
	data = data.sample(n = 20000, random_state = 42)
	data[target_col] = data[target_col].astype(int)
	data.reset_index(drop=True, inplace=True)

	print(data.shape)

	data_config = {
		'target': target_col,
		'important_features_idx': [],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	print(data.shape)
	
	return data, data_config


def process_mimic_mo():

	data = pd.read_csv('data/mimiciii/mimic_mo.csv', low_memory=False)
	target_col = 'mortality'
	data_config = {
		'target': target_col,
		'important_features_idx': [],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	print(data.shape)
	print(data[target_col].value_counts())
	
	return data, data_config

def process_mimic_icd2():

	data = pd.read_csv('data/mimiciii/mimic_icd.csv', low_memory=False)
	target_col ='ICD9_GROUP'
	data_config = {
		'target': target_col,
		'important_features_idx': [],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	print(data.shape)
	print(data[target_col].value_counts())
	
	return data, data_config

def process_mimic_los():

	data = pd.read_csv('data/mimiciii/mimic_los.csv', low_memory=False)
	target_col = 'LENGTH_OF_STAY'
	data_config = {
		'target': target_col,
		'important_features_idx': [],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'regression',
		'data_type': 'tabular'
	}

	print(data.shape)
	
	return data, data_config


def process_genetic(sample = False):
	data = pd.read_csv('data/genetic/clinvar_conflicting.csv', low_memory=False)
	data.drop(["CLNHGVS"],axis = 1, inplace = True)
	target_col = "CLASS"
	data = move_target_to_end(data,target_col)

	def value_correction(df,columns):
	
		def correction(row):
			if pd.isna(row):
				return row
			else:
				if "/" in row:
					return float(row.split("/")[0])/float(row.split("/")[1])
				else:
					first_value = row.split("-")[0]
					if first_value == "?":
						first_value = row.split("-")[1]
					return first_value

		for col in columns:

			df[col] = df[col].apply(correction)
			df[col] = df[col].astype(float)

		return df

	data = value_correction(data,["CDS_position","cDNA_position","Protein_position", "INTRON","EXON"])
	data.drop(["CDS_position","cDNA_position"],axis = 1, inplace = True)
	data['EXON'] = data["EXON"].fillna(data['INTRON'])
	data.drop(["INTRON"], axis = 1, inplace = True)

	# drop missing columns 
	ms_ratio_df = data.isnull().sum()/data.shape[0]
	high_ms_cols = ms_ratio_df[ms_ratio_df>0.95].index
	print(high_ms_cols)
	data = data.drop(high_ms_cols,axis=1)

	# impute missing values
	ms = data.isnull().sum()/data.shape[0]
	ms = ms[ms>0].sort_values(ascending=False)

	cat_missing_cols = [
	'PolyPhen', 'SIFT', 'CLNVI', 'BAM_EDIT', 'Amino_acids', 'Codons', 'MC', 'SYMBOL', 'Feature', 'Feature_type', 'BIOTYPE'
	]

	num_missing_cols = [ col for col in ms.index if col not in cat_missing_cols]

	assert len(cat_missing_cols) + len(num_missing_cols) == len(ms)

	data[cat_missing_cols] = data[cat_missing_cols].fillna(data[cat_missing_cols].mode().iloc[0])
	data[num_missing_cols] = data[num_missing_cols].fillna(data[num_missing_cols].mean())

	# drop rows with missing values
	# data = data.dropna(axis=0)
	# data = data[data['CHROM'] != 'X']
	# data['CHROM'] = data['CHROM'].astype(int)

	# # categorical columns and numerical columns
	cat_cols = [
		'REF', 'ALT', 'CLNDISDB', 'CLNDN', 'CLNVC', 'MC', 'Allele', 'Consequence', 'IMPACT', 'SYMBOL', 'Feature_type',
		'Feature', 'BIOTYPE', 'Amino_acids', 'Codons', 'PolyPhen', 'SIFT',  'CLNVI', 'BAM_EDIT', 'CHROM'
	]

	num_cols = [col for col in data.columns if col not in cat_cols and col != target_col]
	# onehot encoding categorical columns
	assert len(cat_cols) + len(num_cols) + 1 == data.shape[1]

	oh = OneHotEncoder(sparse_output=False, drop="if_binary", max_categories=15)
	oh.fit(data[cat_cols])
	oh_cols = oh.get_feature_names_out()
	oh_data = oh.transform(data[cat_cols])
	oh_data = pd.DataFrame(oh_data, columns=oh_cols, index=data.index)

	print(oh_data.shape)

	# standardize numerical columns
	scaler = StandardScaler()
	scaler.fit(data[num_cols])
	num_data = scaler.transform(data[num_cols])
	num_data = pd.DataFrame(num_data, columns=num_cols, index=data.index)
	print(num_data.shape)

	# combine categorical and numerical columns
	data = pd.concat([oh_data, num_data, data[target_col]], axis=1)
	print(data.shape)

	# pca
	pca = PCA(n_components=0.9)
	pca.fit(data.drop(target_col, axis=1))
	pca_data = pca.transform(data.drop(target_col, axis=1))
	pca_data = pd.DataFrame(pca_data, index=data.index)
	data = pd.concat([pca_data, data[target_col]], axis=1)

	data = convert_gaussian(data, target_col)
	data = normalization(data, target_col)

	if sample:
		data0 = data[data[target_col] == 0]
		data1 = data[data[target_col] == 1]
		if data0.shape[0] > data1.shape[0]:
			data0 = data0.sample(n=data1.shape[0], random_state=0)
		else:
			data1 = data1.sample(n=data0.shape[0], random_state=0)

		data = pd.concat([data0, data1], axis=0)

	if data.shape[0] > 20000:
		data = data.sample(n=20000, random_state=0)

	data_config = {
		'target': target_col,
		'important_features_idx': [],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	print(data.shape)
	print(data[target_col].value_counts())
	
	return data, data_config


#############################################################################################################
# Utilities
#############################################################################################################
def icd_group(x):
    if isinstance(x, str) and (x[0] == 'V'  or x[0] == 'E'):
        if x[0] == 'V':
            return 19
        elif x[0] == 'E':
            return 20
    else:
        icd_int = int(x[0:3])
        if icd_int < 140:
            return 1
        elif icd_int < 240:
            return 2
        elif icd_int < 280:
            return 3
        elif icd_int < 290:
            return 4
        elif icd_int < 320:
            return 5
        elif icd_int < 390:
            return 6
        elif icd_int < 460:
            return 7
        elif icd_int < 520:
            return 8
        elif icd_int < 580:
            return 9
        elif icd_int < 630:
            return 10
        elif icd_int < 680:
            return 11
        elif icd_int < 710:
            return 12
        elif icd_int < 740:
            return 13
        elif icd_int < 780:
            return 14
        elif icd_int < 790:
            return 15
        elif icd_int < 797:
            return 16
        elif icd_int < 800:
            return 17
        else:
            return 18

def convert_df(df):
    df['PAID'] = df['SUBJECT_ID'].astype(str) + '_' + df['HADM_ID'].astype(str)
    df = df[['PAID', 'ITEMID']]
    counts = df.groupby(['PAID', 'ITEMID']).size().reset_index(name='occurrence')
    df = pd.pivot_table(counts, index='PAID', columns='ITEMID', values='occurrence', fill_value=0)
    df = df.reset_index()
    return df.copy()

def convert_pca(df, n_components=20, prefix='patient'):
    df_pca = df.drop('PAID', axis = 1)
    pca = PCA(n_components=n_components)
    pca.fit(df_pca)
    df_pca = pca.transform(df_pca)
    df_pca = pd.DataFrame(df_pca)
    df_pca.columns = [prefix + str(i) for i in range(df_pca.shape[1])]
    df_pca = pd.concat([df['PAID'], df_pca], axis = 1)
    return df_pca.copy()