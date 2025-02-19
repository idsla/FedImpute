
import os
import json
import timeit
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass
import timeit
from tabulate import tabulate
import uuid
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np

from fedimpute.scenario.scenario_builder import ScenarioBuilder
from fedimpute.evaluation.evaluator import Evaluator
from fedimpute.execution_environment.env_fedimp import FedImputeEnv

@dataclass
class FedImputeResult:
    
    # retrieve keys - aspects
    imputer: str
    fed_strategy: str
    imputer_params: dict
    strategy_params: dict
    round_id: int
    
    # values - results
    results: Dict[str, pd.DataFrame]
    run_time_imp: float
    run_time_eval: float


class FedImputePipeline:
    
    def __init__(self):
        
        # pipeline components
        self.experiment_id: Union[str, None] = None
        self.experiment_description: Union[str, None] = None
        self.scenario_builder: Union[ScenarioBuilder, None] = None
        self.fed_imp_configs: List[Dict[str, Union[str, dict]]] = None
        
        # pipeline results
        self.results: List[FedImputeResult] = []
        self.tidy_results: Union[pd.DataFrame, None] = None
        
        # pipeline parameters
        self.repeats: int = 10
        self.seed: int = 100330201
        self.persist_data: bool = False

    def setup(
       self,
       id: str = None,
       fed_imp_configs: List[Tuple[str, str, List[str], dict]] = None,
       evaluation_aspects: List[str] = ['imp_quality', 'local_pred', 'fed_pred'],
       persist_data: bool = False,
       seed: int = 100330201,
       description: str = None,
   ):
        """Initialize pipeline with a scenario"""
        if fed_imp_configs is None:
            fed_imp_configs = self.example_config
       
        self.evaluation_aspects = evaluation_aspects
        if id is not None:
            self.experiment_id = id
        else:
            self.experiment_id = str(uuid.uuid4())
        if description is not None:
            self.experiment_description = description
        else:
            self.experiment_description = ''
        
        self.results = []
        self.persist_data = persist_data
        self.seed = seed
       
        # decompose fed_imp_configs
        configs = []
        for setting in fed_imp_configs:
            imputer, fed_strategies, imputer_params, strategies_params = setting
           
            assert (
                len(fed_strategies) == len(strategies_params), 
                "The number of strategies and strategy parameters must be the same"
            )
            assert (
                isinstance(imputer, str) and isinstance(strategies_params, dict),
                "The imputer must be a string and the strategy parameters must be a dictionary"
            )
            
            for fed_strategy, strategy_params in zip(fed_strategies, strategies_params):
                config = {
                    'imputer': imputer,
                    'fed_strategy': fed_strategy,
                    'imputer_params': imputer_params,
                    'strategy_params': strategy_params,
                }
                configs.append(config)
                
        self.fed_imp_configs = configs
       
    def pipeline_setup_summary(
        self, format: str = 'plain-text', display: bool = True, truncate: bool = True
    ):
        
        if format == 'plain-text':
            line_width = 62
            field_width = 15
            summary_str = '=' * line_width + '\n'
            summary_str += f"Experiment ID: {self.experiment_id}\n"
            summary_str += '=' * line_width + '\n'
            summary_str += f"Description: {self.experiment_description}\n"
            summary_str += f"Persist Data: {self.persist_data}\n"
            summary_str += f"Evaluation: {self.evaluation_aspects}\n"
            summary_str += f"Seed: {self.seed}\n"
            summary_str += '-' * line_width + '\n'
            
            headers = ["Imputer", "Fed Strategy", "Imp Params", "Strategy Params"]
            table_data = []
            for idx, config in enumerate(self.fed_imp_configs):
                max_len = field_width
                if truncate:    
                    table_data.append([
                        config['imputer'], 
                        config['fed_strategy'], 
                        f"{str(config['imputer_params']):.{field_width}s}", 
                        f"{str(config['strategy_params']):.{field_width}s}"
                    ])
                else:
                    table_data.append([
                        config['imputer'], 
                        config['fed_strategy'], 
                        f"{config['imputer_params']}", 
                        f"{config['strategy_params']}"
                    ])
            summary_str += tabulate(table_data, headers=headers, tablefmt="simple", stralign='left', showindex="always") + '\n'
            summary_str += '=' * line_width + '\n'
            if display:
                print(summary_str)
            else:
                return summary_str
        elif format == 'dataframe':
            pass            
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @property
    def example_config(self):
        
        return [
            ('ice', ['local', 'fedice'], {}, {}),
            ('missforest', ['local', 'fedtree'], {}, {}),
        ]
    
    def run_pipeline(
        self, 
        scenario_builder: ScenarioBuilder,
        repeats: int = 5,
        seed: int = 100330201,
        verbose: int = 0
    ):
        """
        Run the pipeline with the given configurations.
        """
        self.repeats = repeats
        self.seed = seed
        self.scenario_builder = scenario_builder
        
        # Decompose the fed_imp_configs
        for idx, setting in enumerate(self.fed_imp_configs):
            imputer = setting['imputer']
            fed_strategy = setting['fed_strategy']
            imputer_params = setting['imputer_params']
            strategy_params = setting['strategy_params']
            
            if verbose > 0:
                print('-'*100)
                print(f"Running experiment: {idx + 1} / {len(self.fed_imp_configs)}")
                print(f"Imputer: {imputer} | Fed Strategy: {fed_strategy} |"
                      f"Imp Params: {imputer_params} | Strategy Params: {strategy_params}")
            
            for repeat in range(self.repeats):            
                seed = self.seed + repeat
                # federated imputation
                start_time = timeit.default_timer()
                env = FedImputeEnv(debug_mode=False)
                env.configuration(
                    imputer = imputer, fed_strategy=fed_strategy, 
                    imputer_params=imputer_params, 
                    fed_strategy_params=strategy_params,
                    seed=seed
                )
                env.setup_from_scenario_builder(scenario_builder = scenario_builder, verbose=0)
                env.run_fed_imputation()
                end_time = timeit.default_timer()
                imputation_time = end_time - start_time
                
                # evaluation
                start_time = timeit.default_timer()
                evaluator = Evaluator()
                evaluator.evaluate_all(env, metrics=self.evaluation_aspects, seed=seed, verbose=0)
                end_time = timeit.default_timer()
                evaluation_time = end_time - start_time
                results = evaluator.export_results(
                    format='dict-dataframe', persist = False, save_path = None
                )
                
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
                
        # convert results to tidy dataframe
        self.tidy_results = self._convert_results_to_tidy_dataframe(self.results)
        
    def reset(self):
        self.__init__()
    
    def display_results(self):
        pass
    
    @staticmethod
    def _convert_results_to_tidy_dataframe(results: List[FedImputeResult]):
        """
        Convert the raw results to a tidy dataframe format.
        """
        if results is None:
            raise ValueError("Current results are empty in the pipeline")
        
        rows = []
        
        for result in results:
            # Base info for this experiment
            base_info = {
                'imputer': result.imputer,
                'fed_strategy': result.fed_strategy,
                'round_id': result.round_id,
            }
            
            # For each evaluation aspect (e.g., imp_quality, pred_downstream_local)
            for aspect, df in result.results.items():
                # Each column is a metric
                for metric_name in df.columns:
                    # Each row represents a client
                    for client_id, value in df[metric_name].items():
                        # metric info
                        row = base_info.copy()
                        row.update({
                            'metric_type': aspect,
                            'metric_name': metric_name,
                            'client_id': client_id,  # client index from DataFrame
                            'value': value
                        })
                        rows.append(row)
        
            # Time info
            row_imp = base_info.copy()
            row_imp.update({
                'metric_type': 'time',
                'metric_name': 'imputation',
                'client_id': 'avg',
                'value': result.run_time_imp
            })
            rows.append(row_imp)
            
            row_eval = base_info.copy()
            row_eval.update({
                'metric_type': 'time',
                'metric_name': 'evaluation',
                'client_id': 'avg',
                'value': result.run_time_eval
            })
            rows.append(row_eval)
            
        return pd.DataFrame(rows)
    
    def show_pipeline_results(
        self, 
        format: str = 'plain-text',
        metric_aspect: str = None,
        metric_name: str = None,
        show_round_variation: bool = False,
        display: bool = True
    ):
        
        data = self._convert_results_to_tidy_dataframe(self.results).copy()
        
        assert (
            (metric_aspect in data['metric_type'].unique()) == True
        ), f"Metric aspect {metric_aspect} not found, supported aspects: {data['metric_type'].unique()}."
        
        # global data filtering and aggregation
        data = data[
            (data['metric_type'] == metric_aspect) & (data['metric_name'] == metric_name)
        ].drop(columns=['metric_type', 'metric_name'])
        
        data = data.groupby(
            ['imputer', 'fed_strategy', 'client_id']
        ).agg({'value': [
            ('value_mean', lambda x: x.mean()), ('value_std', lambda x: x.std() if len(x) > 1 else 0)
        ]}).reset_index()
        
        if format == 'plain-text':
            pass            
        elif format == 'dataframe':
        
            # Create the formatted string of mean(std) values
            data['value_formatted'] = data.apply(
                lambda x: f"{x['value']['value_mean']:.3f} ({x['value']['value_std']:.2f})", 
                axis=1
            )
            
            data['client_id'] = data['client_id'].apply(lambda x: f'Client {x}')
            
            # Pivot the table
            result = data.pivot(
                index='client_id',
                columns=['imputer', 'fed_strategy'],
                values='value_formatted'
            )
            
            result.columns.names = (None, None)
            result.index.name = None
            
            return result.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def plot_pipeline_results(
        self, 
        metric_aspect: str,
        plot_type: str = 'bar',
        plot_params: dict = None,
        save_path: str = None,
    ):
        """
        Plot the pipeline comparison results.
        """
        data = self._convert_results_to_tidy_dataframe(self.results).copy()
        
        assert (
            (metric_aspect in data['metric_type'].unique()) == True
        ), f"Metric aspect {metric_aspect} not found, supported aspects: {data['metric_type'].unique()}."
        
        # global data filtering and aggregation
        data = data[data['metric_type'] == metric_aspect].drop(columns=['metric_type'])
        data_mean = data.groupby(
            ['imputer', 'fed_strategy', 'metric_name', 'client_id']
        ).agg({'value': 'mean'}).reset_index()
        metric_names = data_mean['metric_name'].unique()
        all_strategies = data_mean['fed_strategy'].unique()
        palette = sns.color_palette("Paired", len(all_strategies))
        strategy_colors = {
            strat: palette[i] for i, strat in enumerate(all_strategies)
        }
             
        if plot_params is None:
            plot_params = {
                'font_size': 16,
                'bar_width': 0.2,
            }
        
        if plot_type == 'bar':
            
            if 'bar_width' in plot_params:
                bar_width = plot_params['bar_width']
            else:
                bar_width = 0.4
            
            if 'font_size' in plot_params:
                font_size = plot_params['font_size']
            else:
                font_size = 16
            
            num_bars = data_mean.groupby(['imputer', 'fed_strategy']).ngroups
            num_intervals = data_mean.groupby(['imputer']).ngroups - 1
            interval_width = 0.6
            bar_width = 0.4
            capsize = 2
            plt_height = bar_width * (num_bars+1) + interval_width * num_intervals
            
            n_cols = 4
            if len(metric_names) < n_cols:
                bar_width = bar_width * len(metric_names) / n_cols
                interval_width = interval_width * len(metric_names) / n_cols
                n_cols = len(metric_names)
            n_rows = int(np.ceil(len(metric_names) / n_cols))
            
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(4*n_cols + 0.5*n_cols, plt_height*(n_rows)), sharey=True, squeeze=False
            )
            
            # plot each metric in a separate subplot
            for idx, metric_name in enumerate(metric_names):
                
                ax = axes[idx // n_cols, idx % n_cols]
                
                # data calculation
                metric_data = data_mean[data_mean['metric_name'] == metric_name].drop(columns=['metric_name'])
                metric_data['imputer'] = metric_data['imputer'].str.title()
                
                ################################################################################################################
                # Bar plot
                imputers = metric_data['imputer'].unique()
                current_y = 0
                tick_labels = []
                tick_positions = []
                
                for i, imputer in enumerate(imputers):
                    # data for this imputer
                    imputer_data = metric_data[metric_data['imputer'] == imputer]
                    
                    # bar values, error bar values, positions and colors
                    strategies = imputer_data['fed_strategy'].unique()
                    bar_values, error_bar_values, colors = [], [], []
                    for strat in strategies:
                        bar_value = imputer_data[imputer_data['fed_strategy'] == strat]['value'].mean() 
                        bar_values.append(bar_value)
                        error_bar_values.append(
                            [
                                abs(imputer_data[imputer_data['fed_strategy'] == strat]['value'].min() - bar_value), 
                                abs(imputer_data[imputer_data['fed_strategy'] == strat]['value'].max() - bar_value)
                            ]
                        )
                        colors.append(strategy_colors[strat])

                    bar_values = np.array(bar_values)
                    error_bar_values = np.array(error_bar_values).T
                    
                    positions = [current_y + j*bar_width for j in range(len(strategies))]
                    current_y = positions[-1] + interval_width
                    tick_labels.append(imputer)
                    tick_positions.append((positions[0] + positions[-1])/2)
                    
                    # plot bars
                    ax.barh(
                        y = positions, 
                        width = bar_values, 
                        height = bar_width,
                        linewidth=0,
                        capsize= capsize,
                        error_kw=dict(ecolor='black', alpha = 0.5, lw = 1.5),
                        color = colors,
                        xerr = error_bar_values,
                    )
                
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(tick_labels)
                
                ax.set_title(metric_name, fontsize=font_size, fontweight='bold')
                ax.tick_params(axis='y', labelsize=font_size, labelrotation=45)
                ax.tick_params(axis='x', labelsize=font_size - 2, labelrotation=45)
                for tick in ax.get_yticklabels():
                    tick.set_fontweight('bold')
                for tick in ax.get_xticklabels():
                    tick.set_fontsize(font_size - 2)

            # remove empty axes
            for idx in range(len(metric_names), n_rows * n_cols):
                axes[idx // n_cols, idx % n_cols].set_visible(False)
            
            ################################################################################################################
            # legend
            legend_elements = [
                Patch(facecolor=color, edgecolor='white', label=strat) 
                for strat, color in strategy_colors.items()
            ]
            plt.legend(
                handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1.0), frameon=False, 
                title='Fed Strategy', title_fontproperties={'weight': 'bold', 'size': font_size},  # Bold title
                prop={'weight': 'bold', 'size': font_size - 2}  # Bold labels  
            )
            plt.subplots_adjust(wspace=0.1, hspace=0.3)
            if save_path is not None:
                plt.savefig(save_path, bbox_inches='tight', transparent=True, dpi=300)
            plt.show()
            
        elif plot_type == 'line':
            pass
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
