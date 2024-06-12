from src.modules.data_prep.utils import split_train_test

from src.modules.data_prep.data_prep_his import (
    process_NHIS_income, process_heart, process_skin, process_sepsis,
    process_diabetic, process_diabetic2, process_cardio, process_mimiciii_mortality, process_genetic,
    process_mimiciii_mo2, process_mimic_icd, process_mimic_icd2, process_mimic_mo, process_mimic_los,
    process_breast, process_dermatology, process_pima_diabetes
)

from src.modules.data_prep.data_prep_large import (
    process_ijcnn, process_susy, process_statlog, process_higgs, process_svm
)

from src.modules.data_prep.data_prep_fairness import (
    process_adult, process_default_credit, process_bank_market
)

from src.modules.data_prep.data_prep import (
    process_iris, process_ecoli, process_white, process_red, process_wine_three, process_spam, process_blocks,
    process_glass, process_optdigits, process_segmentation, process_sonar, process_sensor, process_waveform,
    process_yeast, process_letter, process_raisin, process_telugu_vowel, process_telugu_tabular, process_wine,
    process_wifi, process_firewall, process_dry_bean, process_avila, process_pendigits
)

from src.modules.data_prep.data_prep_reg import (
    process_diabetes, process_housing, process_red_reg, process_white_reg
)

from src.modules.data_prep.data_prep_nips import (
    process_hhip, process_codrna, process_california, process_dvisits, process_vehicle, process_codon, process_school
)


def load_data(dataset_name, normalize=True, verbose=False, threshold=None, output_format='dataframe'):

    # ##################################################################################################################
    # Classification
    # ##################################################################################################################
    if dataset_name == 'iris':
        data, data_config = process_iris(normalize, verbose, threshold)
    elif dataset_name == 'breast':
        data, data_config = process_breast(normalize, verbose, threshold)
    elif dataset_name == 'ecoli':
        data, data_config = process_ecoli(normalize, verbose, threshold)
    elif dataset_name == 'white':
        data, data_config = process_white(normalize, verbose, threshold)
    elif dataset_name == 'red':
        data, data_config = process_red(normalize, verbose, threshold)
    elif dataset_name == 'wine_quality_all':
        data, data_config = process_wine_three(normalize, verbose, threshold)
    elif dataset_name == 'spam':
        data, data_config = process_spam(normalize, verbose, threshold)
    elif dataset_name == 'blocks':
        data, data_config = process_blocks(normalize, verbose, threshold)
    elif dataset_name == 'glass':
        data, data_config = process_glass(normalize, verbose, threshold)
    elif dataset_name == 'optdigits':
        data, data_config = process_optdigits(normalize, verbose, threshold)
    elif dataset_name == 'segmentation':
        data, data_config = process_segmentation(normalize, verbose, threshold)
    elif dataset_name == 'sonar':
        data, data_config = process_sonar(normalize, verbose, threshold)
    elif dataset_name == 'sensor':
        data, data_config = process_sensor(normalize, verbose, threshold)
    elif dataset_name == 'sensor_pca':
        data, data_config = process_sensor(normalize, verbose, threshold, pca=True)
    elif dataset_name == 'waveform':
        data, data_config = process_waveform(normalize, verbose, threshold)
    elif dataset_name == 'yeast':
        data, data_config = process_yeast(normalize, verbose, threshold)
    elif dataset_name == 'letter':
        data, data_config = process_letter(normalize, verbose, threshold)
    elif dataset_name == 'raisin':
        data, data_config = process_raisin(normalize, verbose, threshold)
    elif dataset_name == 'telugu_vowel':
        data, data_config = process_telugu_vowel(normalize, verbose, threshold)
    elif dataset_name == 'telugu_tabular':
        data, data_config = process_telugu_tabular(normalize, verbose, threshold)
    elif dataset_name == 'wine':
        data, data_config = process_wine(normalize, verbose, threshold)
    elif dataset_name == 'wifi':
        data, data_config = process_wifi(normalize, verbose, threshold)
    elif dataset_name == 'default_credit':
        data, data_config = process_default_credit(normalize, verbose, threshold)
    elif dataset_name == 'firewall':
        data, data_config = process_firewall(normalize, verbose, threshold)
    elif dataset_name == 'dry_bean':
        data, data_config = process_dry_bean(normalize, verbose, threshold)
    elif dataset_name == 'dry_bean_g':
        data, data_config = process_dry_bean(normalize, verbose, threshold, guassian=True)

    ####################################################################################################################
    # Large Dataset
    ####################################################################################################################
    elif dataset_name == 'adult':
        data, data_config = process_adult(normalize, verbose, threshold)
    elif dataset_name == 'adult_pca':
        data, data_config = process_adult(normalize, verbose, threshold, sample=False, pca=True, gaussian=True)
    elif dataset_name == 'adult_balanced':
        data, data_config = process_adult(normalize, verbose, threshold, sample=True)
    elif dataset_name == 'adult_balanced_pca':
        data, data_config = process_adult(normalize, verbose, threshold, sample=True, pca=True, gaussian=True)
    elif dataset_name == 'bank_marketing':
        data, data_config = process_bank_market(normalize, verbose, threshold)
    elif dataset_name == 'bank_marketing_balanced':
        data, data_config = process_bank_market(normalize, verbose, threshold, sample=True)
    elif dataset_name == 'bank_balanced_pca':
        data, data_config = process_bank_market(normalize, verbose, threshold, sample=True, pca=True, gaussian=True)
    elif dataset_name == 'ijcnn':
        data, data_config = process_ijcnn(normalize, verbose, threshold)
    elif dataset_name == 'ijcnn_balanced':
        data, data_config = process_ijcnn(normalize, verbose, threshold, sample=True)
    elif dataset_name == 'ijcnn_balanced_pca':
        data, data_config = process_ijcnn(normalize, verbose, threshold, sample=True, pca=True, gaussian=True)
    elif dataset_name == 'svm':
        data, data_config = process_svm(normalize, verbose, threshold)
    elif dataset_name == 'svm_g':
        data, data_config = process_svm(normalize, verbose, threshold, gaussian=True)
    elif dataset_name == 'pendigits':
        data, data_config = process_pendigits(normalize, verbose, threshold)
    elif dataset_name == 'pendigits_g':
        data, data_config = process_pendigits(normalize, verbose, threshold, gaussian=True)
    elif dataset_name == 'statlog':
        data, data_config = process_statlog(normalize, verbose, threshold)
    elif dataset_name == 'statlog_pca':
        data, data_config = process_statlog(normalize, verbose, threshold, pca=True, gaussian=True)
    elif dataset_name == 'avila':
        data, data_config = process_avila(normalize, verbose, threshold)
    elif dataset_name == 'susy':
        data, data_config = process_susy(normalize, verbose, threshold)
    elif dataset_name == 'susy_g':
        data, data_config = process_susy(normalize, verbose, threshold, gaussian=True)
    elif dataset_name == 'higgs':
        data, data_config = process_higgs(verbose, threshold)

    ####################################################################################################################
    # Healthcare Dataset
    ####################################################################################################################
    elif dataset_name == 'nhis_income':
        data, data_config = process_NHIS_income(pca=False)
    elif dataset_name == 'nhis_income_pca':
        data, data_config = process_NHIS_income(pca=True)
    elif dataset_name == 'heart':
        data, data_config = process_heart(pca=True, sample=False)
    elif dataset_name == 'heart_balanced':
        data, data_config = process_heart(pca=True, sample=True)
    elif dataset_name == 'skin':
        data, data_config = process_skin(normalize, verbose, threshold, sample=False)
    elif dataset_name == 'skin_balanced':
        data, data_config = process_skin(normalize, verbose, threshold, sample=True)
    # elif dataset_name == 'codrna':
    #     data, data_config = process_codrna(normalize, verbose, threshold, sample=False)
    elif dataset_name == 'codrna_balanced':
        data, data_config = process_codrna(normalize, verbose, threshold, sample=True)
    # elif dataset_name == 'codon':
    #     data, data_config = process_codon(verbose, threshold)
    elif dataset_name == 'sepsis':
        data, data_config = process_sepsis(verbose, threshold)
    elif dataset_name == 'diabetic':
        data, data_config = process_diabetic(verbose, threshold)
    elif dataset_name == 'diabetic_balanced':
        data, data_config = process_diabetic(verbose, threshold, sample=True)
    elif dataset_name == 'diabetic2':
        data, data_config = process_diabetic2(verbose, threshold, sample=True)
    elif dataset_name == 'cardio':
        data, data_config = process_cardio(verbose, threshold)
    elif dataset_name == 'mimiciii_mo':
        data, data_config = process_mimiciii_mortality()
    elif dataset_name == 'mimiciii_icd':
        data, data_config = process_mimic_icd2()
    elif dataset_name == 'mimiciii_mo2':
        data, data_config = process_mimic_mo()
    elif dataset_name == 'mimiciii_los':
        data, data_config = process_mimic_los()
    elif dataset_name == 'genetic':
        data, data_config = process_genetic(sample=False)
    elif dataset_name == 'genetic_balanced':
        data, data_config = process_genetic(sample=True)
    elif dataset_name == 'dermatology':
        data, data_config = process_dermatology(normalize, verbose, threshold)
    elif dataset_name == "pima_diabetes":
        data, data_config = process_pima_diabetes(normalize, verbose, threshold)

    ####################################################################################################################
    # Regression
    ####################################################################################################################
    elif dataset_name == 'diabetes':
        data, data_config = process_diabetes(normalize, verbose, threshold)
    # elif dataset_name == 'california_housing':
    #     data, data_config = process_california_housing(normalize, verbose, threshold)
    elif dataset_name == 'housing':
        data, data_config = process_housing(normalize, verbose, threshold)
    elif dataset_name == 'red_reg':
        data, data_config = process_red_reg(normalize, verbose, threshold)
    elif dataset_name == 'white_reg':
        data, data_config = process_white_reg(normalize, verbose, threshold)
    
    ####################################################################################################################
    # NIPS
    ####################################################################################################################
    elif dataset_name == 'codrna':
        data, data_config = process_codrna(normalize, verbose, threshold, sample=False)
    elif dataset_name == 'hhip':
        data, data_config = process_hhip(verbose)
    elif dataset_name == 'california':
        data, data_config = process_california(verbose)
    elif dataset_name == 'dvisits':
        data, data_config = process_dvisits(verbose)
    elif dataset_name == 'vehicle':
        data, data_config = process_vehicle(verbose)
    elif dataset_name == 'codon':
        data, data_config = process_codon(verbose)
    elif dataset_name == 'school':
        data, data_config = process_school(verbose, pca = False)
    elif dataset_name == 'school_pca':
        data, data_config = process_school(verbose, pca = True)
    else:
        raise Exception("Unknown dataset name {}".format(dataset_name))

    ####################################################################################################################
    # Split train and test
    # if test_size is not None:
    #     train_data, test_data = split_train_test(
    #         data, data_config, test_size=test_size, seed=seed, output_format='dataframe_merge'
    #     )
    #     test_data = test_data.values
    #     train_data = train_data.values
    # else:
    #     train_data = data.values
    #     test_data = None
    
    if output_format == 'dataframe':
        return data, data_config
    elif output_format == 'numpy':
        return data.values, data_config
