# used for recording hyper-parameters
# (num_clients, dataset_name, mechanism_strategy): (gamma, alpha, scale_factor)
PARAMS = {
    "codrna":
        {  # s1 - lr
            (3, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.5, 4),
            (5, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.8, 2),
            (7, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.65, 4),
            (9, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.95, 3),
            (11, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.5, 3),
            # s1 - rl
            (3, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.8, 4),
            (5, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.5, 2),
            (7, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.5, 2),
            (9, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.5, 2),
            (11, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.8, 2),
            # s2 - even
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.0"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.1"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.3"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.5"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.7"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.9"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=1.0"): (None, None, None)
        },
    "codon":
        {  # s1 - lr
            (3, "sample-unevenl1-600", "mnar_lr@sp=extremel1"): (0.05, 0.95, 4),
            (5, "sample-unevenl1-600", "mnar_lr@sp=extremel1"): (0.05, 0.95, 4),
            (7, "sample-unevenl1-600", "mnar_lr@sp=extremel1"): (0.05, 0.95, 4),
            (9, "sample-unevenl1-600", "mnar_lr@sp=extremel1"): (0.05, 0.95, 4),
            (11, "sample-unevenl1-600", "mnar_lr@sp=extremel1"): (0.05, 0.95, 4),
            # s1 - rl
            (3, "sample-unevenr1-600", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            (5, "sample-unevenr1-600", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            (7, "sample-unevenr1-600", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            (9, "sample-unevenr1-600", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            (11, "sample-unevenr1-600", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            # s2 - even
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.0"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.1"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.3"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.5"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.7"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.9"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=1.0"): (None, None, None)
        },
    "heart":
        {  # s1 - lr
            (3, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.95, 4),
            (5, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.95, 4),
            (7, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.95, 4),
            (9, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.95, 4),
            (11, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.95, 4),
            # s1 - rl
            (3, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            (5, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            (7, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            (9, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            (11, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            # s2 - even
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.0"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.1"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.3"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.5"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.7"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.9"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=1.0"): (None, None, None)
        },
    "genetic":
        {  # s1 - lr
            (3, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.95, 4),
            (5, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.95, 4),
            (7, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.95, 4),
            (9, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.95, 4),
            (11, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.95, 4),
            # s1 - rl
            (3, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            (5, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            (7, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            (9, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            (11, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            # s2 - even
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.0"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.1"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.3"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.5"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.7"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.9"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=1.0"): (None, None, None)
        },
    "mimiciii_mo2":
        {  # s1 - lr
            (3, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.5, 4),
            (5, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.8, 4),
            (7, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.8, 4),
            (9, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.8, 4),
            (11, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.5, 4),
            # s1 - rl
            (3, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            (5, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            (7, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.95, 4),
            (9, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.65, 4),
            (11, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.65, 4),
            # s2 - even
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.0"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.1"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.3"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.5"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.7"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.9"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=1.0"): (None, None, None)
        },
    "mimiciii_los":
        {  # s1 - lr
            (3, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.65, 4),
            (5, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.5, 4),
            (7, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.5, 4),
            (9, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.5, 4),
            (11, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.5, 4),
            # s1 - rl
            (3, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.5, 4),
            (5, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.5, 4),
            (7, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.5, 4),
            (9, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.5, 4),
            (11, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.5, 4),
            # s2 - even
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.0"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.1"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.3"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.5"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.7"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.9"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=1.0"): (None, None, None)
        },
    "mimiciii_icd":
        {  # s1 - lr
            (3, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.8, 4),
            (5, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.5, 4),
            (7, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.5, 4),
            (9, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.5, 4),
            (11, "sample-unevenl1-1000", "mnar_lr@sp=extremel1"): (0.05, 0.95, 4),
            # s1 - rl
            (3, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.65, 4),
            (5, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.5, 4),
            (7, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.5, 4),
            (9, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.5, 4),
            (11, "sample-unevenr1-1000", "mnar_lr@sp=extremer1"): (0.05, 0.5, 4),
            # s2 - even
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.0"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.1"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.3"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.5"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.7"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=0.9"): (None, None, None),
            (10, "sample-evenly", "mnar_lr@sp=extreme_r=1.0"): (None, None, None)
        }
}


class Hyperparameters:
    def __init__(self, dataset, num_clients, data_partition, mm_strategy, method):
        self.dataset = dataset
        self.num_clients = num_clients
        self.data_partition = data_partition
        self.mm_strategy = mm_strategy
        self.method = method

        self.params = self._fetch_params(self.method)

    def _fetch_params(self, method):
        if method == 'fedmechw_new':
            try:
                gamma, alpha, beta = PARAMS[self.dataset][(self.num_clients, self.data_partition, self.mm_strategy)]
            except:
                gamma, alpha, beta = None, None, None

            if gamma is None:
                gamma = 0.02
            if alpha is None:
                alpha = 0.95
            if beta is None:
                beta = 4
            if gamma is not None and alpha is not None and beta is not None:
                return {"gamma": gamma, "alpha": alpha, "scale_factor": beta, 'client_thres': 1.0}
            else:
                return None
        else:
            return None

    def get_params(self):
        return self.params

    def __str__(self):
        return (f"({self.dataset}, N={self.num_clients}, {self.data_partition}, {self.mm_strategy}) => "
                f"({self.params})")

