import json

from .backend import FileBackend
from .utils import consolidate_experiment_data
from loguru import logger
import time as time
import multiprocessing as mp


class ExperimentManager:

    def __init__(self, db_backend_option='mongodb'):

        # file backend
        self.file_backend = FileBackend()

        # fed_imp instance
        self.experiment_class = None

    def set_experiment(self, experiment_class):
        self.experiment_class = experiment_class
        logger.info("Experiment name: {}".format(experiment_class.name))

    def run_experiments(
            self, exp_configs, experiment_meta, use_db=False, debug=False
    ):
        experiment_type = experiment_meta['exp_name']
        experiment_id = experiment_meta['exp_id']

        # if use_db:
        #     # connect to backend database
        #     self.backend.connect()
        #     logger.info("Connected to backend database")
        #
        #     # save fed_imp meta data
        #     self.backend.save(experiment_type, experiment_meta)
        #     logger.info("Saved fed_imp meta data")
        # else:
        #     self.file_backend.save(
        #         experiment_meta["dir_name"], experiment_meta["filename"], experiment_meta
        #     )
        #     logger.info("Saved fed_imp meta data")

        # fed_imp start
        logger.info("Running {} experiments".format(len(exp_configs)))
        start_global = time.time()

        # run experiments
        for idx, exp_config in enumerate(exp_configs):
            print("=" * 200)
            logger.info("Experiment - {}".format(idx))
            logger.info("Experiment - {}".format(exp_config.keys))
            logger.info("Experiment - {}".format(exp_config.values))
            start = time.time()

            ############################################################
            # Experiment main process
            # logger.info("config: {}".format(exp_config.config))
            experiment = self.experiment_class(debug=debug)
            with open('config.txt', 'w') as f:
                json.dump(exp_config.config, f)

            ret,_ = experiment.run_experiment(exp_config.config)

            #############################################################
            end = time.time()
            logger.info("Experiment finished in {}".format(end - start))

            if ret is None:
                continue

            # save results to backend
            dir_name, file_name = exp_config.dir_name, exp_config.file_name
            exp_stats = {
                'exp_id': experiment_id,
                'exp_name': experiment_type,
                'time': time.time(),
                'elapsed_time': end - start,
                'file_path': dir_name + '/' + file_name
            }
            experiment_data = consolidate_experiment_data(exp_config, ret, exp_stats)
            self.file_backend.save(dir_name, file_name, experiment_data)

        # total time and summary
        end_global = time.time()
        logger.info("Total time: {}".format(end_global - start_global))

    def run_experiments_mtp(self, exp_configs, experiment_meta, use_db=False, debug=False):
        experiment_type = experiment_meta['exp_name']
        experiment_id = experiment_meta['exp_id']

        self.file_backend.save(
            experiment_meta["dir_name"], experiment_meta["filename"], experiment_meta
        )
        logger.info("Saved fed_imp meta data")

        num_processes = 3
        chunk_size = len(exp_configs) // num_processes
        # chunks = [exp_configs[i:i + chunk_size] for i in range(0, len(exp_configs), chunk_size)]

        # fed_imp start
        logger.info("Running {} experiments parallel".format(len(exp_configs)))
        with mp.Pool(num_processes) as pool:
            process_args = [(idx, exp_config, experiment_type, experiment_id, self.experiment_class, self.file_backend,
                             self.backend, use_db) for idx, exp_config in enumerate(exp_configs)]
            pool.starmap(self.run_chunk_experiment, process_args, chunksize=chunk_size)

    @staticmethod
    def run_chunk_experiment(
            idx, exp_config, experiment_type, experiment_id, experiment_class, file_backend, backend, use_db
    ):
        # run experiments
        print("=" * 200)
        logger.info("Experiment - {}".format(idx))
        logger.info("Experiment - {}".format(exp_config.keys))
        logger.info("Experiment - {}".format(exp_config.values))
        start = time.time()

        ############################################################
        # Experiment main process
        # logger.info("config: {}".format(exp_config.config))
        experiment = experiment_class(debug=False)
        ret = experiment.run_experiment(exp_config.config)

        #############################################################
        end = time.time()
        logger.info("Experiment finished in {}".format(end - start))

        # save results to backend
        dir_name, file_name = exp_config.dir_name, exp_config.file_name
        exp_stats = {
            'exp_id': experiment_id,
            'exp_name': experiment_type,
            'time': time.time(),
            'elapsed_time': end - start,
            'file_path': dir_name + '/' + file_name
        }
        experiment_data = consolidate_experiment_data(exp_config, ret, exp_stats)
        file_backend.save(dir_name, file_name, experiment_data)

        if use_db:
            backend.save(experiment_type, experiment_data)
            logger.info("Experiment saved to backend database")
