import pandas as pd
import os
import json
from config import ROOT_DIR, settings
from datetime import datetime
import glob


def process_ret_to_xlsx(exp_dir, level_dirs, clean_func):

	# collect all results
	ret = collect_all_results(exp_dir, level_dirs, clean_func)

	# save processed results to csv
	df = pd.DataFrame.from_records(ret)
	base = os.path.join(ROOT_DIR, settings['experiment_result_dir'])
	output_dir = os.path.join(base, settings['processed_result_dir'], exp_dir)
	name = "_".join(level_dirs)
	output_filename = os.path.join(output_dir, "results_{}_{}.xlsx".format(name, datetime.now().strftime("%m%d%H%M")))
	print(output_filename)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	df.to_excel(os.path.join(exp_dir, output_filename), index=False)


def collect_all_results(exp_dir, level_dirs, clean_func):
	ret = []
	# expand level directories to access all fed_imp results in the last directory
	base = os.path.join(ROOT_DIR, settings['experiment_result_dir'], settings['raw_result_dir'])
	level_path = os.path.join(base, exp_dir)
	for level_dir in level_dirs:
		level_path = os.path.join(level_path, level_dir)

	# fetch all json files of experiments results
	for file in glob.iglob(level_path + '/**/*.json', recursive=True):
		if not file.startswith("experiment_meta"):
			# read file
			file_path = file
			with open(file_path) as f:
				result = json.load(f)
			# processing json
			cleaned_ret = clean_func(result)
			ret.append(cleaned_ret)

	return ret

