from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

from src.fed_imp.sub_modules.strategy.strategy_imp import StrategyImputation
from src.fed_imp.sub_modules.client.client import Client
from src.fed_imp.sub_modules.model.logistic import LogisticRegression
from src.fed_imp.sub_modules.model.TwoNN import TwoNN
from src.fed_imp.sub_modules.model.utils import init_net
from src.fed_imp.sub_modules.dataloader import construct_tensor_dataset
from typing import Dict, List
from loguru import logger
from torch.utils.data import DataLoader, ConcatDataset
from src.utils import set_seed
from src.fed_imp.sub_modules.server.base_server import ServerBase
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def client_fit(args):
    client, client_id, col_idx = args
    weight, loss, missing_info, projection_matrix, ms_coef = client.fit(
        fit_task='fit_imputation_model', fit_instruction={'feature_idx': col_idx}
    )
    return client_id, weight, loss, missing_info, projection_matrix, ms_coef, client


def process_args(args):
    return client_fit(*args)


class ServerFedAvg(ServerBase):

    def __init__(
            self, clients: Dict[int, Client], strategy_imp: StrategyImputation, server_config: dict,
            pred_config: dict, test_data: np.ndarray = None, seed: int = 21, track=False,
            run_prediction=True, persist_data=False
    ):

        # Basic setup
        super().__init__(
            clients, strategy_imp, server_config, pred_config, test_data, seed, track, run_prediction,
            persist_data
        )

        self.clients = clients
        self.strategy_imp = strategy_imp
        self.max_workers = None
        self.config = server_config
        self.seed = seed
        self.rounds = server_config.get('pred_rounds', 1)
        self.base_model = server_config.get('base_model', 'twonn')

        # imputation parameters
        self.num_rounds_imp = server_config.get('imp_round', 30)
        self.imp_model_fit_mode = server_config.get('model_fit_mode', "one_shot")

        # group clients
        self.client_groups = {}

        ###########################################################################
        # Prediction model
        ###########################################################################
        # model

        other_params = pred_config["model_params"].get('model_other_params', {})
        if other_params is None:
            other_params = {}

        self.model_init_config = pred_config["model_params"].get('model_init_config', {})
        if self.model_init_config is None:
            self.model_init_config = {}

        # training parameters
        self.pred_config = pred_config
        self.batch_size = pred_config["train_params"].get('batch_size', 128)
        self.learning_rate = pred_config["train_params"].get('learning_rate', 0.001)
        self.weight_decay = pred_config["train_params"].get('weight_decay', 1e-4)
        self.num_rounds_pred = pred_config["train_params"].get('pred_round', 1000)
        self.pred_local_epochs = pred_config["train_params"].get('pred_local_epochs', 5)


        # test data
        self.test_data = test_data
        self.test_dataset = construct_tensor_dataset(self.test_data[:, :-1], self.test_data[:, -1])
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        self.validate_data = None
        self.validate_dataset = None
        self.val_dataloader = None

    def prediction(self, seeds=None):
        ###############################################################################################
        # N rounds Final FedAvg federated prediction model learning
        ###############################################################################################
        # setup work in client side
        if seeds is None:
            seeds = list(range(42, 42 + self.rounds))

        # validation data
        validate_datas = []
        for client_id, client in self.clients.items():
            client.local_dataset()
            validate_datas.append(client.val_data)

        self.validate_data = np.concatenate(validate_datas, axis=0)
        self.validate_dataset = construct_tensor_dataset(self.validate_data[:, :-1], self.validate_data[:, -1])
        self.val_dataloader = DataLoader(self.validate_dataset, batch_size=self.batch_size, shuffle=False)

        # construct training dataloader
        for client_id, client in self.clients.items():
            client.pred_data_setup(self.batch_size)

        ################################################################################################
        # Prediction model 1
        ################################################################################################
        # initialize initial global model and dispatch it to all clients
        # torch.manual_seed(1000)
        best_accus, best_f1s, histories = [], [], []

        for s in seeds:

            set_seed(s)

            # self.pred_model = init_net(self.pred_model, **self.model_init_config)
            if self.base_model == 'twonn':
                pred_model = TwoNN(
                    in_features=self.test_data.shape[1] - 1, num_classes=len(np.unique(self.test_data[:, -1])),
                    num_hiddens=self.pred_config['model_params']['num_hiddens']
                )
            elif self.base_model == 'lr':
                pred_model = LogisticRegression(
                    in_features=self.test_data.shape[1] - 1, num_classes=len(np.unique(self.test_data[:, -1]))
                )
            else:
                raise ValueError('base model not supported')

            # N epochs of federated training
            clients_prediction_history = []
            best_accu, counter, patience = 0, 0, 300
            for current_round in range(1, self.num_rounds_pred + 1):
                test_loss, test_accu, test_f1, val_loss, val_accu, val_f1 = self._run_round_prediction(
                    pred_model, server_round=current_round,
                )

                clients_prediction_history.append(
                    {
                        'test_loss': test_loss, 'test_accu': test_accu, 'test_f1': test_f1,
                        'val_loss': val_loss, 'val_accu': val_accu, 'val_f1': val_f1
                    }
                )

                if current_round % 50 == 0:
                    logger.info(
                        'Round: {}, test_accu: {:.4f}, test_f1: {:.4f}, val_loss: {:.4f}, val_accu: {:.4f}, val_f1: {'
                        ':.4f}'.format(
                            current_round, test_accu, test_f1, val_loss, val_accu, val_f1
                        )
                    )

                if current_round >= 150:
                    if test_accu >= best_accu:
                        best_accu = test_accu
                        counter = 0
                    else:
                        counter += 1

                    if counter >= patience:
                        logger.info("Early stop at round {}".format(current_round))
                        break

                # Test the model
            accu_rounds_average = [item['test_accu'] for item in clients_prediction_history]
            f1_rounds_average = [item['test_f1'] for item in clients_prediction_history]
            best_accus.append(np.max(accu_rounds_average))
            best_f1s.append(np.max(f1_rounds_average))
            histories.append(clients_prediction_history)

        # logger.info(
        #     "model1 test acc: {:.6f} ({:.3f}), test f1: {:.6f} ({:.3f})".format(
        #         np.array(best_accus).mean(), np.array(best_accus).std(), np.array(best_f1s).mean(),
        #         np.array(best_f1s).std()
        #     )
        # )

        return best_accus, best_f1s

    ####################################################################################################################
    # Prediction
    ####################################################################################################################
    def _run_round_prediction(self, pred_model, server_round):

        if server_round == 1:
            self.transmit_model(pred_model, init=False)
        else:
            self.transmit_model(pred_model, init=False)

        # Clients local training
        self.pred_local_training()

        # average local training results
        self.average_model(pred_model)

        # evaluation
        val_loss, val_acc, val_f1 = self.evaluate_pred_model(pred_model, validate=True)
        test_loss, test_acc, test_f1 = self.evaluate_pred_model(pred_model, validate=False)

        return test_loss, test_acc, test_f1, val_loss, val_acc, val_f1

    ####################################################################################################################
    # Prediction Utils function
    ####################################################################################################################
    def transmit_model(self, pred_model, init=False):
        for idx, client in self.clients.items():
            if init:
                torch.manual_seed(self.seed + idx)
                pred_model = init_net(pred_model, **self.model_init_config)
            client.pred_model = deepcopy(pred_model)

    def pred_local_training(self):
        """Update local model using local dataset."""
        for _, client in self.clients.items():
            client.local_train_pred(self.pred_local_epochs, self.learning_rate, self.weight_decay)

    def average_model(self, pred_model):
        """Average the updated and transmitted parameters from each selected client."""
        averaged_weights = OrderedDict()
        clients = list(self.clients.values())
        sample_sizes = [client.get_sample_size() for client in clients]
        normalized_coefficient = [size / sum(sample_sizes) for size in sample_sizes]
        for it, client in enumerate(clients):
            local_weights = client.pred_model.state_dict()
            for key in pred_model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = normalized_coefficient[it] * local_weights[key]
                else:
                    averaged_weights[key] += normalized_coefficient[it] * local_weights[key]
        pred_model.load_state_dict(averaged_weights)

    def evaluate_pred_model(self, pred_model, validate=False):
        pred_model.eval()
        pred_model.to(DEVICE)

        if validate:
            test_dataloader = self.val_dataloader
            X_test, y_test = self.validate_data[:, :-1], self.validate_data[:, -1]
        else:
            X_test, y_test = self.test_data[:, :-1], self.test_data[:, -1]
            test_dataloader = self.test_dataloader

        with torch.no_grad():
            test_loss, counter = 0, 0
            for data, labels in test_dataloader:
                counter += 1
                data, labels = data.float().to(DEVICE), labels.long().to(DEVICE)
                outputs = pred_model(data)
                test_loss += torch.nn.CrossEntropyLoss()(outputs, labels).item()

                # f1 score
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

        pred_model.to(DEVICE)
        test_epoch_loss = test_loss / counter
        outputs = pred_model(torch.FloatTensor(X_test).to(DEVICE))
        _, predicted = torch.max(outputs.data, 1)
        test_accuracy = accuracy_score(y_test, predicted.to('cpu').numpy())
        test_f1 = f1_score(y_test, predicted.to('cpu').numpy(), average='macro')

        return test_epoch_loss, test_accuracy, test_f1
