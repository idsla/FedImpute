import pandas as pd
import json

scenarios = ['ideal', 'random', 's1', 's2', 's3', 's4']
methods = ['central2', 'local', 'fedavg-s', 'cafe']
root = 'results/raw_results/fed_imp_pred_fed/codon/sample-evenly/'
ret = []

for scenario in scenarios:
    for method in methods:
        with open(root + f'/{scenario}/{method}_fedavg_mlp_pytorch_pred.json') as f:
            content = json.load(f)
            ret.append({
                'scenario': scenario,
                'method': method,
                'accu': content['results']["accu_mean"],
                'f1': content['results']["f1_mean"],
                'roc': content['results']["roc_mean"],
                'prc': content['results']["prc_mean"],
            })
            
df = pd.DataFrame(ret)
df = df.replace('central2', 'central')
df = df.replace('fedavg-s', 'savg')

df.to_excel('results/processed_results/results_fedpred.xlsx', index=False)