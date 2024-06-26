"""
main script for running experiments
"""
from src.experiment_management.experiment_manager import ExperimentManager
from src.fed_imp.experiment import Experiment
from src.experiment_management.load_config import load_configs_raw
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="conf", config_name="exp_config_imp")
def main(cfg: DictConfig) -> None:
	print(OmegaConf.to_yaml(cfg))
	debug = cfg.debug
	# debug mode
	if debug:
		logger.remove(0)
		logger.add(sys.stderr, level="DEBUG")
	else:
		logger.remove(0)
		logger.add(sys.stderr, level="INFO")

	# initialize fed_imp manager and fed_imp class
	experiment_manager = ExperimentManager()
	experiment_manager.set_experiment(Experiment)

	# load configuration files and fed_imp type
	configs, exp_meta = load_configs_raw(OmegaConf.to_container(cfg, resolve=True))

	# run experiments and persist results
	experiment_manager.run_experiments(configs, exp_meta, use_db=False, debug=debug)


if __name__ == '__main__':
	main()
