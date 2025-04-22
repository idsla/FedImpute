# Pipeline API for Benchmarking

To facilitate the systematic evaluation of multiple algorithms, the fedimpute framework provides a streamlined pipeline execution interface through the `fedimpute.Pipeline` class. This pipeline enables concurrent execution and evaluation of multiple algorithms, supporting
comprehensive benchmarking studies and comparative analyses. We briefly introduce core functions below. 

## Pipeline Setup and Execution

The pipeline workflow begins with a constructed distributed data scenario. Users first need to instantiate an object from `fedimpute.Pipeline` class. Then, they need to provide the configuration of the algorithms to execute in the pipeline through the `setup()` method, which accepts two primary parameters:

- `fed_imp_configs`: a list of configuration tuples, where each tuple specifies an imputer, associated federated
aggregation strategies, and their parameters. 
- `evaluation_aspects`: a list of evaluation criteria including imputation quality (`imp_quality`), local prediction (`local_pred`), and federated prediction (`fed_pred`).
- `evaluation_params`: a dictionary to specify the evaluation process. It includes `metrics` a list specifying the aspects of evaluation and `model` which downstream task used for evaluation.

Finally, the pipeline execution is via the `run_pipeline()` method, which takes a scenario object as input. It executes all configured algorithms on the input scenario, performs evaluations, and stores the results.

We also have a `pipeline_setup_summary()` function to provide a summary of constructed pipeline.

```{python}
from fedimpute.pipeline import FedImputePipeline

pipeline = FedImputePipeline()
pipeline.setup(
    id = 'benchmark_demo',
    fed_imp_configs = [
        ('em', ['local', 'fedem'], {}, [{}, {}]),
        ('mice', ['local', 'fedmice'], {}, [{}, {}]),
        ('gain', ['local', 'fedavg'], {}, [{}, {}]),
    ],
    evaluation_params = {
        'metrics': ['imp_quality', 'local_pred', 'fed_pred'],
        'model': 'lr',
    },
    persist_data = False,
    description = 'benchmark demonstration'
)

pipeline.pipeline_setup_summary()

pipeline.run_pipeline(
    scenario_builder, repeats = 5, verbose = 0
)
```

## Result Analysis and Visualization
The pipeline provides two key functions for helping analyze the pipeline execution results:

- `show_pipeline_results()` which generates tabular summaries of specific metrics across different algorithms and strategies. 

- `plot_pipeline_results()` which creates comparative visualizations of performance metrics across different imputation and federated aggregation strategies.

```{python}
pipeline.plot_pipeline_results(metric_aspect = 'fed_pred_personalized', plot_type = 'bar')
```