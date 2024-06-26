import pandas as pd
import json

scenarios = ['ideal', 's1', 's2', 's3', 's4', 'complex1', 'complex2']
methods = ['central2', 'local', 'fedavg-s', 'cafe']
root = 'results/raw_results/fed_imp/codon/sample-evenly/'
ret = []

for scenario in scenarios:
    for method in methods:
        with open(root + f'{scenario}/as_{method}@s_102931466@s_50@p_False.json') as f:
            content = json.load(f)
            ret.append({
                'scenario': scenario,
                'method': method,
                'rmse': content['results']["avg_rets_final"]["imp@rmse"],
                'sliced-ws': content['results']["avg_rets_final"]["imp@sliced_ws"]
            })
            
df = pd.DataFrame(ret)
df = df.replace('central2', 'central')
df = df.replace('fedavg-s', 'savg')

df.to_excel('results/processed_results/results_fedimp.xlsx', index=False)
