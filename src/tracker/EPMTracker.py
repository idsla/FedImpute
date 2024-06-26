from dataclasses import dataclass, field
from typing import List, Union

import numpy as np


@dataclass
class EMPRecord:
    iteration: int
    feature_idx: int
    aggregation_weights: Union[np.ndarray, None]
    global_model_params: Union[np.ndarray, None]
    local_imp_model_params: Union[np.ndarray, None]
    local_mm_model_params: Union[np.ndarray, None]
    losses: Union[np.ndarray, None]


@dataclass
class ClientInfo:

    client_id: int
    missing_ratio: float = 0.0
    missing_mask: np.ndarray = None
    data_true: np.ndarray = None
    data_imp: List[np.ndarray] = field(default_factory=lambda: [])


class EMPTracker:
    def __init__(self):
        self.records = []
        self.imp_quality = []
        self.client_infos = []

    def add_record(self, record: EMPRecord):
        self.records.append(record)

    def fetch_records_feature(self, feature_idx):
        return [record for record in self.records if record.feature_idx == feature_idx]

    def fetch_records_iteration(self, iteration):
        return [record for record in self.records if record.iteration == iteration]
