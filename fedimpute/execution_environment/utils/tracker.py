from typing import List, Union, Dict, Tuple, Optional
import numpy as np
import math
import io
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
class Tracker:
    """
    Tracker class to track the imputation results along iterations
    tracker_params: {
        "track_data": bool,          # whether to track imputed data along iterations
        "track_model_params": bool,  # whether to track imputation model parameters along iterations
        "track_misc": bool,          # whether to track other parameters along iterations
        "persist": str - one of {'none', 'final', 'all'}  # options persist imputed data and model parameters
    }
    """

    def __init__(self, tracker_params: dict = None):
        
        self.writer = SummaryWriter(log_dir='./.logs/')

        if tracker_params is None:
            tracker_params = {
                "track_data": False,
                "track_model_params": False,
                "track_misc": False,
                "persist": 'none'
            }

        # options
        if 'track_data' not in tracker_params:
            self.track_data = False
        else:
            assert isinstance(tracker_params['track_data'], bool), "track_data is not a boolean"
            self.track_data = tracker_params['track_data']

        if 'track_misc' not in tracker_params:
            self.track_misc = False
        else:
            assert isinstance(tracker_params['track_misc'], bool), "track_misc is not a boolean"
            self.track_misc = tracker_params['track_misc']

        if 'persist' not in tracker_params:
            self.persist = False
        else:
            self.persist = tracker_params['persist']

        # internal data structures
        self.num_clients = None
        self.rounds = []
        self.imp_quality = []  # tracking history results of imputation quality
        self.imp_data = []  # tracking final imputed data
        self.model_params = []  # tracking final imputation model parameters
        self.other_info = []  # tracking other parameters
        self.misc = []  # tracking other parameters

        self.origin_data = None  # tracking original data
        self.mask = None  # tracking missing mask
        self.split_indices = None  # tracking split indices
        self.imp_data_final = None  # tracking final imputed data

    def record_initial(self, data: List[np.ndarray], mask: List[np.ndarray], imp_quality: dict):

        # self.origin_data = np.concatenate(data)
        # self.mask = np.concatenate(mask)
        # self.split_indices = np.cumsum([item.shape[0] for item in data])[:-1]
        self.num_clients = len(data)
        self.rounds.append(0)
        self.imp_quality.append(imp_quality)
        self.other_info.append(None)

    def record_round(
            self, round_num: int, imp_quality: dict,
            data: List[np.ndarray], model_params: List[dict], other_info: List[dict]
    ):

        self.rounds.append(round_num)
        self.imp_quality.append(imp_quality)
        self.other_info.append(other_info)
        
        if imp_quality is not None:
            for key, value in imp_quality.items():
                self.writer.add_scalars(key, {f"client_{k}": v for k, v in value.items()}, round_num)
        
        if other_info is not None:
            for key, value in other_info.items():
                self.writer.add_scalars(key, {f"client_{k}": v for k, v in value.items()}, round_num)

        # if self.track_misc and other_info is not None:
        #     self.misc.append(other_info)

    def record_final(
            self,
            imp_quality: dict,
            data: List[np.ndarray], model_params: List[dict], other_info: List[dict]
    ):

        self.rounds.append(len(self.rounds) + 1)
        self.imp_quality.append(imp_quality)
        self.other_info.append(other_info)
        self.writer.close()

        # if self.track_misc and other_info is not None:
        #     self.misc.append(other_info)

    def to_dict(self) -> dict:

        ret = {
            "results": {
                'rounds': self.rounds,
                "imp_quality": self.imp_quality,
                "other_info": self.other_info,
            }
        }

        if self.persist:
            raise NotImplementedError("Final persist is not implemented yet")
        else:
            ret['persist'] = {}

        return ret
    
    def visualize_imputation_process(
        self, 
        n_cols: int = 5, 
        display: bool = True,
        fontsize: int = 12,
        dpi: int = 150
    ):
        
        rounds = self.rounds
        df_imp = self.processing_tracking_metric(self.imp_quality[:-1], self.num_clients, self.rounds)
        df_other = self.processing_tracking_metric(self.other_info[:-1], self.num_clients, self.rounds)
        df = pd.concat([df_imp, df_other], axis=0)
        if len(df) == 0:
            print("No data in imputation progress to visualize.")
            return None
        
        # visualize
        num_metrics = len(df['metric'].unique())
        num_clients = len(df['client'].unique())
        x_min = df['round'].min() - 0.2
        x_max = df['round'].max() + 0.2
        
        # Set consistent colors for each client
        colors = sns.color_palette(n_colors=num_clients)
        client_colors = {f'client_{i}': colors[i] for i in range(num_clients)}
        sns.set_palette(colors)
        
        if num_metrics <= n_cols:
            n_cols = num_metrics
            
        n_rows = math.ceil(num_metrics / n_cols)
        sns.set_theme(style = "darkgrid")
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), dpi=dpi, squeeze=False)
        
        for i, metric_name in enumerate(df['metric'].unique()):
            ax = axs[i // n_cols, i % n_cols]
            
            df_metric = df[df['metric'] == metric_name]
            
            sns.lineplot(
                x='round', y='value', data=df_metric, hue='client', ax=ax, 
                legend=False, palette=client_colors,
                markers=True, marker='o', markersize=5, linewidth=2
            )
            
            ax.set_title(metric_name.title(), fontsize=fontsize, fontweight='bold')
            ax.set_xlabel('Round', fontsize=fontsize, fontweight='bold')
            ax.set_ylabel('', fontsize=fontsize)
            ax.set_xlim(x_min, x_max)
        
        plt.tight_layout()
        if display:
            plt.show()
        else:
            plt.close()
            return None
            # export figure to string with base64
            # buf = io.BytesIO()
            # plt.savefig(buf, format='png')
            # buf.seek(0)
            # plt.close()
            # return base64.b64encode(buf.getvalue()).decode('utf-8')

        
    
    @staticmethod
    def processing_tracking_metric(
        metric_value_array: List[Dict[str, dict]], 
        num_clients: int, 
        rounds: List[int]
    ) -> pd.DataFrame:
        
        # get all metrics
        metrics = Tracker.get_metrics_from_results(metric_value_array)
        
        # long format dataframe
        df_records = []

        # processing metric value array to format of metrics_progress
        for round_idx, round_value in enumerate(rounds[:-1]):
            for metric_name in metrics:
                for client_idx in range(num_clients):
                    try:
                        df_records.append({
                            'round': round_value,
                            'client': f'client_{client_idx}',
                            'metric': metric_name,
                            'value': metric_value_array[round_idx][metric_name][client_idx]
                        })
                    except:
                        df_records.append({
                            'round': round_value,
                            'client': f'client_{client_idx}',
                            'metric': metric_name,
                            'value': None
                        })
        
        return pd.DataFrame(df_records)
                    
    
    @staticmethod
    def get_metrics_from_results(results: List[Dict[str, dict]]) -> List[str]:
        
        metrics = set()
        for ret in results:
            if ret is None:
                continue
            
            for metric_name, metric_value in ret.items():
                metrics.add(metric_name)
        
        return list(metrics)
        
        
        
        
