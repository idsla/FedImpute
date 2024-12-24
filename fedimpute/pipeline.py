
import os
import json
import timeit
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass
import timeit


from fedimpute.scenario.simulator import Simulator
from fedimpute.evaluation.evaluator import Evaluator
from fedimpute.execution_environment.env_fedimp import FedImputeEnv

@dataclass
class FedImputeResult:
    imputer: str
    fed_strategy: str
    imputer_params: dict
    strategy_params: dict
    round_id: int
    results: dict
    run_time_imp: float
    run_time_eval: float


class FedImputePipeline:
    
    def __init__(self):
        
        # pipeline components
        self.experiment_id: Union[str, None] = None
        self.scenario_simulator: Union[Simulator, None] = None
        
        # pipeline results
        self.results: List[FedImputeResult] = []
        
        # pipeline parameters
        self.repeats: int = 10
        self.seed: int = 100330201
        self.persist_data: bool = False

    def setup(
       self,
       scenario_simulator: Simulator,
       experiment_id: str = None,
       persist_data: bool = False,
       seed: int = 100330201
   ):
       """Initialize pipeline with a scenario"""
       self.scenario_simulator = scenario_simulator
       self.experiment_id = experiment_id
       self.results = []
       self.persist_data = persist_data
       self.seed = seed
    
    @property
    def example_config(self):
        
        return [
            ('ice', ['local', 'fedice'], {}, {}),
            ('missforest', ['local', 'fedtree'], {}, {}),
        ]
    
    def run_pipeline(
        self, 
        fed_imp_configs: List[Tuple[str, str, List[str], dict]],
        evaluation_aspects: List[str] = ['imp_quality', 'local_pred', 'fed_pred'],
        repeats: int = 10,
        seed: int = 100330201,
    ):
        """
        Run the pipeline with the given configurations.
        """
        self.repeats = repeats
        self.seed = seed
        
        if fed_imp_configs is None:
            fed_imp_configs = self.example_config
        
        # Decompose the fed_imp_configs
        for setting in fed_imp_configs:
            imputer, fed_strategies, imputer_params, strategy_params = setting
            for fed_strategy in fed_strategies:
                for repeat in range(self.repeats):            
                    seed = self.seed + repeat
                    # federated imputation
                    start_time = timeit.default_timer()
                    env = FedImputeEnv(debug_mode=False)
                    env.configuration(
                        imputer = imputer, fed_strategy=fed_strategy, 
                        imputer_params=imputer_params, 
                        strategy_params=strategy_params,
                        seed=seed
                    )
                    env.setup_from_simulator(simulator = self.scenario_simulator, verbose=0)
                    env.run_fed_imputation()
                    end_time = timeit.default_timer()
                    imputation_time = end_time - start_time
                    
                    # evaluation
                    start_time = timeit.default_timer()
                    evaluator = Evaluator()
                    evaluator.evaluate_all(env, metrics=evaluation_aspects, seed=seed, verbose=0)
                    end_time = timeit.default_timer()
                    evaluation_time = end_time - start_time
                    
                    results = evaluator.results
                    
                    # save results
                    result = FedImputeResult(
                        imputer=imputer, 
                        fed_strategy=fed_strategy, 
                        imputer_params=imputer_params, 
                        strategy_params=strategy_params,
                        round_id=repeat, 
                        results=results,
                        run_time_imp=imputation_time,
                        run_time_eval=evaluation_time
                    )
                    self.results.append(result)
        
    def save_results(self):
        pass
    
    def compare_results(self):
        pass
    
    def plot_results(self):
        pass
    
    def display_results(self):
        pass
