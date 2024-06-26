from datetime import datetime
import itertools
from .config_manager.config_manage import (
	get_exp_file_name, get_exp_config, ExperimentConfig,
	load_config_fields_from_dicts,
)
from hashlib import md5


def load_configs_raw(config: dict):

	experiment_name = config['experiment_name']
	config_tmpl = config['config_tmpl']
	vars_config = config['config_vars']
	# Experiment ID
	# --------------------------------------------------------------------------------------------
	exp_id = md5("{}{}".format(experiment_name, str(datetime.now().strftime("%m%d%H"))).encode()).hexdigest()

	# Experiment Marker
	# --------------------------------------------------------------------------------------------
	if "experiment_marker" not in config:
		experiment_marker = []
	else:
		experiment_marker = config['experiment_marker']
	if isinstance(experiment_marker, list):
		experiment_marker = process_experiment_marker(experiment_marker, config_tmpl)

	# Experiment Configs
	# --------------------------------------------------------------------------------------------
	exp_configs = []
	for key, value in vars_config.items():

		experiment_vars = load_config_fields_from_dicts(value)

		# generate all combinations
		vars_values = [item.value for item in experiment_vars]
		combinations = list(
			itertools.product(*vars_values)
		)

		# generate configs and corresponding file names
		for combination in combinations:
			# copy config
			config = get_exp_config(config_tmpl, combination, experiment_vars)

			# generate file and dir names
			dir_name, file_name = get_exp_file_name(combination, experiment_vars)
			dir_name = f"fed_imp/{dir_name}"

			# append to exp_configs
			exp_config = ExperimentConfig(
				config=config, experiment_type=experiment_name, dir_name=dir_name,
				file_name=file_name, values=combination, keys=[item.abbr_name for item in experiment_vars]
			)
			exp_configs.append(exp_config)

	# Experiment Meta
	# --------------------------------------------------------------------------------------------
	exp_meta = {
		"exp_id": exp_id,
		"exp_name": experiment_name,
		"dir_name": "{}/{}".format(experiment_name, experiment_marker, datetime.now().strftime("%m%d")),
		"filename": "experiment_meta@{}.json".format(datetime.now().strftime("%m%d%H%M")),
		"type": "meta",
		"date": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
		"variations": vars_config,
		"config_tmpl": config_tmpl
	}

	return exp_configs, exp_meta


def process_experiment_marker(experiment_marker: list, config_tmpl: dict) -> str:
	ret_dirs = []
	for item in experiment_marker:
		# process each entry
		items = item.split("@")
		entry_name = '-'.join(''.join([ele[0] for ele in element.split('_')]) for element in items)
		value = config_tmpl[items[0]]
		for key in items[1:]:
			value = value[key]

		if isinstance(value, list):
			raise NotImplementedError  # TODO
		elif isinstance(value, int):
			ret_dirs.append(entry_name + str(value))
		else:
			ret_dirs.append(entry_name + "_" + str(value))

	return "@".join(ret_dirs)
