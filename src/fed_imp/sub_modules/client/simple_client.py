import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.modules.evaluation.evaluation import Evaluator
import numpy as np
from src.fed_imp.sub_modules.dataloader import construct_tensor_dataset
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
import random
from src.utils import set_seed

class SimpleClient:
    """
    Client class for federated learning
    """

    def __init__(
            self,
            client_id,
            data_true,
            data_imp,
            missing_mask,
            data_test,
            seed: int = 21,
            imbalance=None,
            regression=False,
    ):
        ################################################################################################################
        # Data
        ################################################################################################################
        self.X_train_filled, self.y_train_filled = data_imp[:, :-1], data_imp[:, -1]
        self.client_id = client_id
        self.X_train, self.y_train = data_true[:, :-1], data_true[:, -1]
        self.X_test, self.y_test = data_test[:, :-1], data_test[:, -1]
        self.missing_mask = missing_mask
        self.regression = regression

        # print(self.X_train_filled.shape, self.y_train_filled.shape)
        # print(self.X_train.shape, self.y_train.shape, self.X_test.shape, self.y_test.shape)

        if imbalance is not None:
            if imbalance == 'smote':
                sm = SMOTE(random_state=seed)
                self.X_train_filled, self.y_train_filled = sm.fit_resample(self.X_train_filled, self.y_train_filled)
            elif imbalance == 'smotetm':
                sm = SMOTETomek(random_state=seed)
                # sm = RandomOverSampler(random_state=seed)
                self.X_train_filled, self.y_train_filled = sm.fit_resample(self.X_train_filled, self.y_train_filled)
            elif imbalance == 'smoteenn':
                sm = SMOTEENN(random_state=seed)
                # sm = RandomOverSampler(random_state=seed)
                self.X_train_filled, self.y_train_filled = sm.fit_resample(self.X_train_filled, self.y_train_filled)
            elif imbalance == 'rus':
                sm = RandomUnderSampler(random_state=seed)
                # sm = RandomOverSampler(random_state=seed)
                self.X_train_filled, self.y_train_filled = sm.fit_resample(self.X_train_filled, self.y_train_filled)
            elif imbalance == 'ros':
                # sm = RandomUnderSampler(random_state = 42)
                sm = RandomOverSampler(random_state=seed)
                self.X_train_filled, self.y_train_filled = sm.fit_resample(self.X_train_filled, self.y_train_filled)
            else:
                raise ValueError('Invalid imbalance method')

        ################################################################################################################
        # prediction
        ################################################################################################################
        self.seed = seed
        self.pred_model = None
        self.train_data = None
        self.val_data = None

        ################################################################################################################
        # split training and validation data
        ################################################################################################################

        self.train_data = np.concatenate((self.X_train_filled, self.y_train_filled.reshape(-1, 1)), axis=1)
        self.val_data = None
        self.local_pred_dataset = None
        self.val_data_loader = None
        self.train_dataloader = None
        self.local_pred_dataset_val = None

    ####################################################################################################################
    # prediction model helper functions
    ####################################################################################################################
    def local_train_pred(self, local_epoch, lr, weight_decay):

        pred_model = self.pred_model

        # pred model1
        set_seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred_model.train()
        pred_model.to(device)

        optimizer = torch.optim.Adam(pred_model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

        train_running_loss = []
        for e in range(local_epoch):
            train_epoch_loss, counter = 0, 0
            for data, labels in self.train_dataloader:
                counter += 1
                if self.regression:
                    data, labels = data.float().to(device), labels.float().to(device).view(-1, 1)
                else:
                    data, labels = data.float().to(device), labels.long().to(device)

                optimizer.zero_grad()
                outputs = pred_model(data)
                if self.regression:
                    loss = torch.nn.MSELoss()(outputs, labels)
                else:
                    loss = torch.nn.CrossEntropyLoss()(outputs, labels)

                loss.backward()
                optimizer.step()

                if device == "cuda":
                    torch.cuda.empty_cache()

                train_epoch_loss += loss.item()

            train_epoch_loss /= counter
            train_running_loss.append(train_epoch_loss)

        pred_model.to("cpu")

        return np.array(train_running_loss).mean()

    def get_sample_size(self):
        return self.train_data.shape[0]

    def pred_data_setup(self, batch_size):

        set_seed(self.seed)
        self.local_pred_dataset = construct_tensor_dataset(self.train_data[:, :-1], self.train_data[:, -1])
        self.local_pred_dataset_val = None

        g = torch.Generator()
        g.manual_seed(self.seed)

        self.train_dataloader = DataLoader(
            self.local_pred_dataset, batch_size=batch_size, shuffle=True, generator=g)
        self.val_data_loader = None
