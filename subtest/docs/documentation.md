# Distributed learning contributivity
# Work in progress

## Summary
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
  * [My first scenario](#my-first-scenario)
  * [Select a pre-implemented dataset](#select-a-pre-implemented-dataset)
  * [Set some ML parameters](#set-some-ml-parameters)
  * [Run it](#run-it)
  * [Results](#results)
  * [Contributivity measurement methods](#contributivity-measurement-methods)
- [Scenario parameters](#scenario-parameters)
  * [Choice of dataset](#choice-of-dataset)
  * [Definition of collaborative scenarios](#definition-of-collaborative-scenarios)
  * [Configuration of the collaborative and distributed learning](#configuration-of-the-collaborative-and-distributed-learning)
  * [Configuration of contributivity measurement methods to be tested](#configuration-of-contributivity-measurement-methods-to-be-tested)
  * [Miscellaneous](#miscellaneous)
- [Dataset generation](#dataset-generation)
  * [Dataset](#dataset)
  * [Model generator](#model-generator)
  * [Preprocessing](#preprocessing)
  * [Train/validation/test splits](#train-validation-test-splits)
- [Contacts, contributions, collaborations](#contacts--contributions--collaborations)

## Prerequisites

At root folder:
```bash
pip install -r requirements.txt
pip install -i https://test.pypi.org/simple/ subtest==0.0.0.18
```
Note: This is the temporary package for our library.

## Quick start

### My first scenario
To launch a collaborative round, you need to generate scenario and run it, though a Scenario object.

There are 2 mandatory parameters for a collaborative run: `partners_count` and `amounts_per_partner`.
For instance, you could want to see what is happening with 3 partners, the first with 20% of the total dataset, the second 50% and the third 30% (for a total of 100%).
Here is an example of how you should do it:
```python
from subtest.scenario import Scenario
my_scenario = Scenario(partners_count=3,
                       amounts_per_partner=[0.2, 0.3, 0.5])
```
Note that you can use more advanced sample split options in order to fine tune the data distribution between partners. See the doc

At this point, you can already launch your first scenario !
### Select a pre-implemented dataset
You might also want to consider other parameters such as the dataset, for instance. The easiest way to select a dataset is to use those which are already implemented in subtest. 
Currently MNIST, CIFAR10, TITANIC and ESC50 are handled. You can use one of those by simply passing the parameter dataset_name to your scenario object
```python
from subtest.scenario import Scenario
my_scenario = Scenario(partners_count=3,
                       amounts_per_partner=[0.2, 0.3, 0.5],
                        dataset_name='mnist')
```
With each dataset, a model is provided, so you do not need to care of it. Moreover, the split between the validation and train sets is done by the constructor's of the dataset, even if you can fine thune it.
If you want to use an homemade dataset or a homemade model, you will have to use the [dataset class.](#dataset-generation)
Note that this parameter is not mandatory as the MNIST dataset is selected by default. 

### Set some ML parameters
Even if default training values are provided, it is strongly advised to adapt these to your case. 
For instance you can want your training to go for 10 epochs and 3 minibatches per epoch. 
```python
from subtest.scenario import Scenario
my_scenario = Scenario(partners_count=3,
                       amounts_per_partner=[0.2, 0.3, 0.5],
                       dataset_name='mnist',
                       epoch_count=10,
                       minibatch_count=3)
```
### Run it
You scenario is ready, you can run it.
```python
my_scenario.run()
```

### Results

After a run, every information regarding the training phase will be available under the `multi_partner_learning` object in the `scenario.dataset.mpl` attribute

For instance: you can access `scenario.mpl.loss_collective_models`.

Here is a non exhaustive list of metrics available:

- `loss_collective_models`
- `score_matrix_collective_models`
- `score_matrix_per_partner`

Here is an example of an accuracy plot for 3 partner
```python
import pandas as pd
import seaborn as sns
sns.set()

x = my_scenario.mpl.score_matrix_per_partner

x_collective = my_scenario.mpl.score_matrix_collective_models

x = x[:,:,0]
x_collective = x_collective[:,0]


d = {
    'partner 0' : x[:,0],
    'partner 1' : x[:,1],
    'partner 2' : x[:,2],
    'Averaged model' : x_collective
}

df = pd.DataFrame(d)


sns.relplot(data = df, kind = "line")

```

Check out our [Tutorial 3](https://github.com/SubstraFoundation/distributed-learning-contributivity/blob/master/notebooks/examples/3_Exploring_results.ipynb) for more information.

### Contributivity measurement methods

To use contributivity measurement tools, you will have to change the parameters of your Scenario object
```python
from subtest.scenario import Scenario
my_scenario = Scenario(partners_count=3,
                       amounts_per_partner=[0.2, 0.3, 0.5],
                       dataset_name='mnist',
                       epoch_count=10,
                       minibatch_count=3,
                       methods=['Shapley values'])
```
The result's access is straightforward
```python
contributivity_score = my_scenario.contributivity_list
print(contributivity_score[0])
```

Check out our [Tutorial 4](https://github.com/SubstraFoundation/distributed-learning-contributivity/blob/master/notebooks/examples/4_Exploring_contributivity.ipynb) for more information.

There is a lot more parameters that you can play with, which are fully explained below, in the documentation

## Scenario parameters

### Choice of dataset
There is two way to select a dataset. You can either choice a pre-implemented dataset, by setting the `dataset_name` parameter, or directly pass the dataset object to the `dataset` parameter. To look at the structure of the dataset object, see the [related documentation](#dataset-generation)

`dataset` : `None` (default), `datasets.Dataset object`. If None, the dataset provided by the `dataset_name` will be used
`dataset_name`: `'mnist'` (default), `'cifar10'`, `'esc50'` or `'titanic'`
MNIST, CIFAR10, ESC50 and Titanic are currently supported. They come with their associated modules in `/datasets` for loading data, pre-processing inputs, and define a model architecture.\
For each dataset, it is possible to provide a path to model weights learned from a previous coalition. Use `'random_initialization'` if you want a random initialization or an empty value as in one of the two following syntaxes:
You can also use your own dataset, with the class Dataset. The [Tutorial 2](https://github.com/SubstraFoundation/distributed-learning-contributivity/blob/master/notebooks/examples/2%20_Sentiment140.ipynb) provides more explicit information.

**Note on validation and test datasets**:

- The dataset modules must provide separated train and test sets (referred to as global train set and global test set).
- The global train set is then further split into a global train set and a global validation set.
In the multi-partner learning computations, the global validation set is used for early stopping and the global test set is used for performance evaluation.
- The global train set is split amongst partner (according to the scenario configuration) to populate the partner's local datasets.
- For each partner, the local dataset can be split into separated train, validation and test sets, depending on the dataset configuration. Currently, the local validation and test set are not used, but they are available for further developments of multi-partner learning and contributivity measurement approaches.

`dataset_proportion`: `float` (default: `1`)
This argument allows you to make computation on a sub-dataset of the provided dataset.
This is the proportion of the dataset (initially the train and test sets) which is randomly selected to create a sub-dataset,
it's done before the creation of the global validation set.
You have to ensure that `0 < dataset_proportion <= 1`

### Definition of collaborative scenarios

`partners_count`: `int`  
Number of partners in the mocked collaborative ML scenario.  
Example: `partners_count: 4`

`amounts_per_partner`: `[float]`  
Fractions of the original dataset each partner receives to mock a collaborative ML scenario where each partner provides data for the ML training.  
You have to ensure the fractions sum up to 1.
Example: `amounts_per_partner: [0.3, 0.3, 0.1, 0.3]`

`samples_split_option`: `['basic', 'random']` (default), `['basic', 'stratified']` or `['advanced', [[nb of clusters (int), 'shared' or 'specific']]]`   
How the original dataset data samples are split among partners:

- `'basic'` approaches:
  - `'random'`: the dataset is shuffled and partners receive data samples selected randomly
  - `'stratified'`: the dataset is stratified per class and each partner receives certain classes only (note: depending on the `amounts_per_partner` specified, there might be small overlaps of classes)
- `'advanced'` approach `[[nb of clusters (int), 'shared' or 'specific']]`: in certain cases it might be interesting to split the dataset among partners in a more elaborate way. For that we consider the data samples from the initial dataset as split in clusters per data labels. The advanced split is configured by indicating, for each partner in sequence, the following 2 elements:
  - `nb of clusters (int)`: the given partner will receive data samples from that many different clusters (clusters of data samples per labels/classes)
  - `'shared'` or `'specific'`:
    - `'shared'`: all partners with option `'shared'` receive data samples picked from clusters they all share data samples from
    - `'specific'`: each partner with option `'specific'` receives data samples picked from cluster(s) it is the only one to receive from

Example: `['advanced', [[7, 'shared'], [6, 'shared'], [2, 'specific'], [1, 'specific']]]`

![Example of the advanced split option](https://github.com/SubstraFoundation/distributed-learning-contributivity/blob/master/imgadvanced_split_example.png)

`corrupted_datasets`: `[not_corrupted (default), shuffled or corrupted]`  
Enables to artificially corrupt the data of one or several partners:

- `not_corrupted`: data are not corrupted
- `shuffled`: labels are shuffled randomly, not corresponding anymore with inputs
- `corrupted`: labels are all offseted of `1` class

Example: `[not_corrupted, not_corrupted, not_corrupted, shuffled]`

### Configuration of the collaborative and distributed learning

There are several parameters influencing how the collaborative and distributed learning is done over the datasets of the partners. The following schema introduces certain definitions used in the below description of parameters:

![Schema epochs mini-batches gradient updates](https://github.com/SubstraFoundation/distributed-learning-contributivity/blob/master/imgepoch_minibatch_gradientupdates.png)

`multi_partner_learning_approach`: `'fedavg'` (default), `'seq-pure'`, `'seq-with-final-agg'` or `'seqavg'`  
Define the multi-partner learning approach, among the following as described by the schemas:

- `'fedavg'`: stands for federated averaging

    ![Schema fedavg](https://github.com/SubstraFoundation/distributed-learning-contributivity/blob/master/imgcollaborative_rounds_fedavg.png)

- `'seq-...'`: stands for sequential and comes with 2 variations, `'seq-pure'` with no aggregation at all, and `'seq-with-final-agg'` where an aggregation is performed before evaluating on the validation set and test set (on last mini-batch of each epoch) for mitigating impact when the very last subset on which the model is trained is of low quality, or corrupted, or just detrimental to the model performance.

    ![Schema seq](https://github.com/SubstraFoundation/distributed-learning-contributivity/blob/master/imgcollaborative_rounds_seq.png)

- `'seqavg'`: stands for sequential averaging

    ![Schema seqavg](https://github.com/SubstraFoundation/distributed-learning-contributivity/blob/master/imgcollaborative_rounds_seqavg.png)

Example: `multi_partner_learning_approach: 'seqavg'`

`aggregation_weighting`: `'uniform'` (default), `'data_volume'` or `'local_score'`  
After a training iteration over a given mini-batch, how individual models of each partner are aggregated:

- `'uniform'`: simple average (non-weighted)
- `'data_volume'`: average weighted with per the amounts of data of partners (number of data samples)
- `'local_score'`: average weighted with the performance (on a central validation set) of the individual models

Example: `aggregation_weighting = 'data_volume'`

`epoch_count`: `int` (default: `40`)  
Number of epochs, i.e. of passes over the entire datasets. Superseded when `is_early_stopping` is set to `true`.  
Example: `epoch_count = 30`

`minibatch_count`: `int` (default: `20`)  
Within an epoch, the learning on each partner's dataset is not done in one single pass. The partners' datasets are split into multiple *mini-batches*, over which learning iterations are performed. These iterations are repeated for all *mini-batches* into which the partner's datasets are split at the beginning of each epoch. This gives a total of `epoch_count * minibatch_count` learning iterations.  
Example: `minibatch_count = 20`

`gradient_updates_per_pass_count`: `int` (default: `8`)  
The ML training implemented relies on Keras' `.fit()` function, which takes as argument a `batch_size` interpreted by `fit()` as the number of samples per gradient update. Depending on the number of samples in the train dataset, this defines how many gradient updates are done by `.fit()`. The `gradient_updates_per_pass_count` parameter enables to specify this number of gradient updates per `.fit()` iteration (both in multi-partner setting where there is 1 `.fit()` iteration per mini-batch, and in single-partner setting where there is 1 `.fit()` iteration per epoch).  
Example: `gradient_updates_per_pass_count = 5`

`is_early_stopping`: `True` (default) or `False`  
When set to `True`, the training phases (whether multi-partner of single-partner) are stopped when the performance on the validation set reaches a plateau.  
Example: `is_early_stopping = False`

**Note:** to only launch the distributed learning on the scenarios (and no contributivity measurement methods), simply omit the `methods` parameter (see section [Configuration of contributivity measurement methods to be tested](#configuration-of-contributivity-measurement-methods-to-be-tested) below).

### Configuration of contributivity measurement methods to be tested

`methods`:  
A declarative list `[]` of the contributivity measurement methods to be executed.
All methods available are:

```
- "Shapley values"
- "Independent scores"
- "TMCS"
- "ITMCS"
- "IS_lin_S"
- "IS_reg_S"
- "AIS_Kriging_S"
- "SMCS"
- "WR_SMC"
```

See below section [Contributivity measurement approaches studied and implemented](#contributivity-measurement-approaches-studied-and-implemented) for explanation of the different methods.  
**Note:** When `methods` is omitted in the config file only the distributed learning is run.  
Example: `["Shapley values", "Independent scores", "TMCS"]`

### Miscellaneous

`is_quick_demo`: `True` or `False` (default)  
When set to `True`, the amount of data samples and the number of epochs and mini-batches are significantly reduced, to minimize the duration of the run. This is particularly useful for quick demos or debugging.  
Example: `is_quick_demo = True`

## Dataset generation

The dataset object is useful if you want to define your dataset and relatives objects such as preprocessing functions and model generator.

### Dataset

This is the structure of the dataset generator:

```python
dataset = dataset.Dataset(
    "name",
    X_train,
    X_test,
    y_train,
    y_test,
    input_shape,
    num_classes,
    preprocess_dataset_labels,      # See below
    generate_new_model_for_dataset  # See below
    train_val_split_global,         # See below
    train_test_split_local,         # See below
    train_val_split_local           # See below
)
```

### Model generator
This function provides the model use, which will be trained by the scenario object. 

```python
def generate_new_model_for_dataset():
    model = Sequential()
    # add layers
    model.add(Dense(num_classes, activation='softmax'))
    # compile with loss and accuracy
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```
Note: It is mandatory to have loss and accuracy as metrics for your model.

### Preprocessing
```python
def preprocess_dataset_labels(y):
    # Do stuff
    return y
```

### Train/validation/test splits

The dataset object must be provided separated train and test sets (referred to as global train set and global test set).
The global train set is then further split into a global train set and a global validation set, by the function `train_val_split_global`. Please denote that if this function is not provided, the sklearn's train_test_split function will be called by default, and 10% of the training set will be use as validation set. 
In the multi-partner learning computations, the global validation set is used for early stopping and the global test set is used for performance evaluation.
The global train set is then split amongst partners (according to the scenario configuration) to populate the partner's local datasets.
For each partner, the local dataset will be split into separated train, validation and test sets, using the `train_test_split_local` and `train_val_split_local` functions.
These are not mandatory, by default the local dataset will not be split. 
Denote that currently, the local validation and test set are not used, but they are available for further developments of multi-partner learning and contributivity measurement approaches.



## Contacts, contributions, collaborations
Should you be interested in this open effort and would like to share any question, suggestion or input, you can use the following channels:

- This Github repository (issues or PRs)
- Substra Foundation's Slack workspace, channel #workgroup-mpl-contributivity
- Email: hello@substra.org
- Come meet with us at La Paillasse (Paris, France), Le Palace (Nantes, France) or Studio Iconosquare (Limoges, France)


*Work in progress*
