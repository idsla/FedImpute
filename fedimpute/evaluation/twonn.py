from copy import deepcopy

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import loguru
from ..utils.nn_utils import EarlyStopping
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
from ..utils.reproduce_utils import set_seed

class TwoLayerNNBase(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNNBase, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class TwoNNRegressor(nn.Module):
    def __init__(
            self, 
            hidden_size=32, 
            epochs=1000, 
            lr=0.001, 
            batch_size=64, 
            early_stopping_rounds=30,
            weight_decay=0.00, 
            tol=0.0001, 
            log_interval=10, 
            optimizer='sgd', 
            batch_norm=True,
            dropout=0.0
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.weight_decay = weight_decay
        self.tol = tol
        self.log_interval = log_interval

        self.network = None
        self.dataset = None
        self.dataloader = None
        self.criterion = nn.MSELoss()
        self.optimizer = optimizer
        self.batch_norm = batch_norm
        self.dropout = dropout

    def _build_network(self, input_size, seed = 0):
        set_seed(seed)
        self.hidden_size = input_size*2
        if self.batch_norm:
            self.network = nn.Sequential(
                nn.Linear(input_size, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, 1)
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(input_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, 1)
            )
        
        self.network.apply(init_weights)

    def forward(self, X):
        return self.network(X)

    def fit(self, X, y, seed = 0):

        set_seed(0)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Ensure y_tensor is 2D for MSE Loss

        # Prepare dataset for DataLoader
        if self.dataset is None:
            self.dataset = TensorDataset(X_tensor, y_tensor)

            if len(X) < self.batch_size:
                self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            else:
                self.dataloader = DataLoader(
                    self.dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=True
                )

        # Build the network on first call to fit
        if self.network is None:
            self._build_network(input_size=X.shape[1], seed = seed)

        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=0, momentum=0)

        #self.network
        self.network.to(DEVICE)
        self.train()
        final_loss = 0
        early_stopping = EarlyStopping(
            tolerance=self.tol, tolerance_patience=self.early_stopping_rounds,
            increase_patience=self.early_stopping_rounds, window_size=1, check_steps=1, backward_window_size=1
        )

        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0
            for X_batch, y_batch in self.dataloader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = self(X_batch)
                loss = self.criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(self.dataloader)
            final_loss = avg_epoch_loss

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            early_stopping.update(avg_epoch_loss)
            if early_stopping.check_convergence():
                loguru.logger.debug(f"Early stopping at epoch {epoch}")
                break

            if epoch % self.log_interval == 0:
                loguru.logger.debug(f'Epoch {epoch + 1}/{self.epochs}, Loss: {avg_epoch_loss}')

        return {
            'loss': final_loss,
            'sample_size': len(X),
        }

    def validate(self, X_val, y_val, seed = 0):

        set_seed(seed)
        self.eval()
        val_loss_choice = 'rmse'
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
            outputs = self(X_val_tensor)
            loss = self.criterion(outputs, y_val_tensor)
            val_loss = loss.item()
            
            if val_loss_choice == 'rmse':
                # Calculate RMSE
                rmse = torch.sqrt(torch.mean((outputs - y_val_tensor) ** 2)).item()
                val_loss = rmse
            elif val_loss_choice == 'mae':
                # Calculate MAE
                mae = torch.mean(torch.abs(outputs - y_val_tensor)).item()
                val_loss = mae

        return val_loss
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self(torch.tensor(X, dtype=torch.float32))
        return outputs.numpy().flatten()

    def get_parameters(self):
        return self.network.state_dict()

    def update_parameters(self, new_params):
        model_params = self.network.state_dict()
        model_params.update(new_params)
        self.network.load_state_dict(new_params)
        return self


# Example Usage:
# regressor = TwoNNRegressor(input_size=10, hidden_size=50)
# regressor.fit(X_train, y_train, epochs=50, lr=0.001, batch_size=64)
# predictions = regressor.predict(X_test)


class TwoNNClassifier(nn.Module):
    def __init__(
            self, 
            hidden_size=32, 
            epochs=1000, 
            lr=0.001, 
            batch_size=64, 
            early_stopping_rounds=30, 
            weight_decay=0,
            tol=0.0001, 
            log_interval=10, 
            optimizer='sgd', 
            batch_norm=True,
            dropout=0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.weight_decay = weight_decay
        self.tol = tol
        self.log_interval = log_interval
        self.network = None
        self.dataset = None
        self.dataloader = None
        self.optimizer = optimizer
        self.batch_norm = batch_norm
        self.dropout = dropout

    def _build_network(self, input_size, output_size, class_weight, seed = 0):
        
        if class_weight is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weight)
        
        set_seed(seed)
        self.hidden_size = input_size*2
        if self.batch_norm:
            self.network = nn.Sequential(
                nn.Linear(input_size, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, output_size)
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(input_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, output_size)
            )
        
        self.network.apply(init_weights)

    def forward(self, x):
        return self.network(x)

    def fit(self, X, y, seed = 0):

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        #class_weight = calculate_class_weights(y)       

        # Prepare dataset for DataLoader
        if self.dataset is None:
            self.dataset = TensorDataset(X_tensor, y_tensor)

            if len(X) < self.batch_size:
                self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            else:
                self.dataloader = DataLoader(
                    self.dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=True
                )

        # Determine the number of unique classes to set output size
        if self.network is None:
            unique_classes = np.unique(y)
            self._build_network(input_size=X.shape[1], output_size=len(unique_classes), class_weight=None, seed = seed)

        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=0, momentum=0)

        self.network.to(DEVICE)
        self.train()
        final_loss = 0
        early_stopping = EarlyStopping(
            tolerance=self.tol, tolerance_patience=self.early_stopping_rounds,
            increase_patience=self.early_stopping_rounds, window_size=1, check_steps=1, backward_window_size=1
        )
        for epoch in range(self.epochs):
            self.train()
            epoch_loss = 0
            for X_batch, y_batch in self.dataloader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = self(X_batch)
                loss = self.criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(self.dataloader)
            final_loss = avg_epoch_loss

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            early_stopping.update(avg_epoch_loss)
            if early_stopping.check_convergence():
                loguru.logger.info(f"Early stopping at epoch {epoch}")
                break

            if epoch % self.log_interval == 0:
                loguru.logger.debug(f'Epoch {epoch + 1}/{self.epochs}, Loss: {avg_epoch_loss}')

        self.network.to('cpu')

        return {
            'loss': final_loss,
            'sample_size': len(X),
        }
    
    def validate(self, X_val, y_val, seed = 0):

        set_seed(seed)
        self.eval()
        val_loss_choice = 'f1'
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(DEVICE)
            outputs = self(X_val_tensor)
            loss = self.criterion(outputs, y_val_tensor)
            val_loss = loss.item()
            
            # Calculate additional validation metrics
            _, predicted = torch.max(outputs.data, 1)
            
            if val_loss_choice == 'f1':
                # Convert tensors to numpy for sklearn metrics
                y_true = y_val_tensor.cpu().numpy()
                y_pred = predicted.cpu().numpy()
                # Calculate F1 score (macro average)
                from sklearn.metrics import f1_score
                f1 = f1_score(y_true, y_pred, average='macro')
                val_loss = 1 - f1  # Convert to loss (lower is better)
            
            elif val_loss_choice == 'accuracy':
                # Calculate accuracy
                correct = (predicted == y_val_tensor).sum().item()
                accuracy = correct / y_val_tensor.size(0)
                val_loss = 1 - accuracy  # Convert to loss (lower is better)

        self.network.to('cpu')
        return val_loss

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self(torch.tensor(X, dtype=torch.float32))
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self(torch.tensor(X, dtype=torch.float32))
            probabilities = nn.functional.softmax(outputs, dim=1)
        return probabilities.numpy()

    def get_parameters(self):
        return deepcopy(self.network.state_dict())

    def update_parameters(self, new_params):
        self.network.load_state_dict(new_params)
        return self


def calculate_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Calculate class weights inversely proportional to the class frequencies.

    Parameters:
    y (array-like): The target labels for the training dataset.

    Returns:
    torch.Tensor: The weights for each class.
    """

    # Convert y to a tensor if it isn't one already
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long)

    # Get the number of instances for each class
    class_counts = torch.bincount(y)

    # Compute the inverse of each count
    # Avoid division by zero by adding a small epsilon where there are no instances
    epsilon = 1e-9
    inverse_weights = 1.0 / (class_counts + epsilon)

    # Normalize the weights so that they sum to 1
    normalized_weights = inverse_weights / inverse_weights.sum()

    return normalized_weights
