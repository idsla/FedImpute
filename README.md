<p align="center"><img src="docs/icon.jpg" width="400" height="240"></p>
<h2 align='center'> FedImpute: a benchmarking and evaluation tool for federated imputation across various missing data scenarios. </h2>

<div align="center">
    
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Docs site](https://img.shields.io/badge/docs-GitHub_Pages-blue)](https://idsla.github.io/FedImpute/)

</div>

FedImpute is a benchmarking tool for the evaluation of federated imputation algorithms over various missing data scenarios under horizontally partitioned data.

- **Documentation:** [Documentation](https://idsla.github.io/FedImpute/)
- **Source Code:** [Source Code](https://github.com/idsla/FedImpute/)
- **Tutorial:** [Tutorials] (https://github.com/idsla/FedImpute/tutorials/)

## Installation
Firstly, install python >= 3.10.0, we have two ways to install

Install from pip:
```bash
pip install fedimpute
```

Install from package repo:
```bash
git clone https://github.com/idsla/FedImpute
cd FedImpute

python -m venv ./venv

# window gitbash
source ./venv/Scripts/activate

# linux/unix
source ./venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```
## Basic Usage

**See documentation:** [Documentation](https://idsla.github.io/FedImpute/)

## Supported Algorithms

Imputation Algorithms:

|     Method     |     Type      |               Fed Strategy               |  Imputer (code)  | Reference                                                                                                                                                                                   |
|:--------------:|:-------------:|:----------------------------------------:|:----------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|    Mean    |    Non-NN     |                    `local`, `fedmean`                     |    `mean`    | -                                                                                                                                                                                           |
|     EM     |    Non-NN     |                    `local`, `fedem`                     |     `em`     | [EM](https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_EM.py), [FedEM](https://arxiv.org/abs/2108.10252)                                     |
|    MICE     |    Non-NN     |                  `local`, `fedmice`                     |    `mice`     | [FedICE](https://pubmed.ncbi.nlm.nih.gov/33122624/)                                                                                                                                         |
| MissForest |    Non-NN     |       `local`, `fedtree`                    | `missforest` | [MissForest](https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_missforest.py), [Fed Randomforest](https://pubmed.ncbi.nlm.nih.gov/35139148/) |
|     MIWAE      |      NN       |    `local`, `fedavg`, `fedprox`, ...    |     `miwae`      | [MIWAE](https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_miwae.py)                                                                          |
|      GAIN      |      NN       |     `local`, `fedavg`, `fedprox`, ...     |      `gain`      | [GAIN](https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/imputers/plugin_gain.py)                                                                            |
|     Not-MIWAE      |      NN       |     `local`, `fedavg`, `fedprox`, ...     |     `notmiwae`      | [Not-MIWAE](https://arxiv.org/abs/2006.12871)
|     GNR      |      NN       |     `local`, `fedavg`, `fedprox`, ...     |     `gnr`      | [GNR](https://dl.acm.org/doi/abs/10.1145/3583780.3614835?casa_token=o8dv16sHJcMAAAAA:aAIvug_7cp9oUJSB7ZfTvzUksPyuP6Jbcl3TlHsvXXGEwIe4AbQuHCTlxXZtjDKlymfO30n2o-E9iw)

Federated Strategies:

|   Method   |      Type       | Fed_strategy(code) | Reference      |
|:----------:|:---------------:|:------------------:|:---------------|
|   Local   |    non-federated    |      `local`      | -     |
|   FedMean   |    traditional    |      `fedmean`      | -     |
|   FedEM   |    traditional    |      `fedem`      | [FedEM]()     |
|   FedMICE   |    traditional    |      `fedmice`      | [FedMICE]()     |
|   FedTree   |    traditional    |      `fedtree`      | [FedTree]()     |
|   FedAvg   |    global FL    |      `fedavg`      | [FedAvg](https://arxiv.org/pdf/1602.05629)     |
|  FedProx   |    global FL    |     `fedprox`      | [FedProx](https://arxiv.org/pdf/1812.06127)    |
|  Scaffold  |    global FL    |     `scaffold`     | [Scaffold](https://arxiv.org/pdf/1910.06378)   |
|  FedAdam   |    global FL    |     `fedadam`      | [FedAdam](https://arxiv.org/pdf/2003.00295)    |
| FedAdagrad |    global FL    |    `fedadagrad`    | [FedAdaGrad](https://arxiv.org/pdf/2003.00295) |
|  FedYogi   |    global FL    |     `fedyogi`      | [FedYogi](https://arxiv.org/pdf/2003.00295)    |
| FedAvg-FT  | personalized FL |    `fedavg_ft`     | [FedAvg-FT]()  |


## Contact
For any questions, please contact [Sitao Min](mailto:sm2370@rutgers.edu)
