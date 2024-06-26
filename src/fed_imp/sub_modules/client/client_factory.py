from .client import Client
from .client_vae import ClientVAE
from .client_gain import ClientGAIN


class ClientsFactory:

    def __init__(self, debug=False):
        self.debug = debug

    def generate_clients(self, num_clients, data_partitions, data_ms_clients, test_data, data_config, imputation_config, seed=201030,
                         client_type='ice'):

        clients = {}
        for i in range(num_clients):
            client_dict = {
                'client_id': i,
                'client_data': {
                    'train_data': data_partitions[i],
                    'train_data_ms': data_ms_clients[i],
                    'test_data': test_data,
                },
                'client_data_config': data_config,
                "imputation_config": imputation_config,
                "debug": self.debug,
                'seed': seed + i * 10089
            }
            client_id = client_dict['client_id']
            if client_type == 'ice':
                clients[client_id] = Client(**client_dict)
            elif client_type == 'vae':
                clients[client_id] = ClientVAE(**client_dict)
            elif client_type == 'gain':
                clients[client_id] = ClientGAIN(**client_dict)
            else:
                raise ValueError(f'Invalid client type: {client_type}')
            clients[client_id] = Client(**client_dict)

        return clients


if __name__ == '__main__':
    pass
    # # test add missing
    # data = np.arange(25).reshape(5, 5).astype(float)
    # print(data)
    # missing_strategy = {0: 1, 1: 0.2, 2: 0.4, 3: 0, 4: 0}
    # print(add_missing(data, missing_strategy))
