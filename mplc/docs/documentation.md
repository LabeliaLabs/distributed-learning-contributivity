# Distributed learning contributivity

*Work in progress*

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

You need to install mplc. All the dependencies will be installed automatically. 

```bash
$ pip install mplc
```

This installs the last packaged version on pypi.

If you want to install mplc from the repository, make sure that you got the latest version of pip. 
Then clone the repository, and trigger the installation using pip.

```bash
$ git clone https://github.com/SubstraFoundation/distributed-learning-contributivity.git
$ cd distributed-learning-contributivity
$ pip install -e . 
```


## Quick start

First a few words of context! This library enables to run a multi-partner learning and contributivity measurement experiment. This breaks down into three relatively independent blocks:

1. Creating a mock collaborative multi-partner learning scenario
1. Running a multi-partner ML algorithm to learn a model on all partners respective 
1. Running one or several contributivity measurement methods to evaluate the performance contribution of each partner's dataset to the model performance.

### My first scenario

To run a multi-partner learning and contributivity measurement experiment, you have to define the scenario of your experiment. For that you'll use the `Scenario` object, in which you will define: 

- what dataset will be used and how it will be partitioned among the partners
- what multi-partner learning approach will be used, with what options
- what contributivity measurement approach(es) will be run

There are only 2 mandatory parameters to define a scenario: `partners_count` and `amounts_per_partner`. Many more exist but are optional as default values are configured. You can browse them all in below section [Scenario parameters](#scenario-parameters).

For this very first scenario, you could for example want to see what is happening with 3 partners, where the first one gets 20% of the total dataset, the second one 50% and the third one 30% (for a total of 100%!):

```python
from mplc.scenario import Scenario

my_scenario = Scenario(partners_count=3,
                       amounts_per_partner=[0.2, 0.3, 0.5])
```

Note that you can use more advanced partitioning options in order to finetune the data distribution between partners to your likings.

At this point, you can already launch your first scenario ! But before showing you the `run()` button, let's go just a bit further.

### Select a pre-implemented dataset

You might also want to consider other parameters such as the dataset to be used, for instance. The easiest way to select a dataset is to use those which are already implemented in subtest. 
Currently MNIST, CIFAR10, TITANIC, IMDB and ESC50 are supported. You can use one of those by simply passing the parameter dataset_name to your scenario object

```python
from mplc.scenario import Scenario
my_scenario = Scenario(partners_count=3,
                       amounts_per_partner=[0.2, 0.3, 0.5],
                        dataset_name='mnist')
```
With each dataset, a model is provided, so you do not need to care of it. Moreover, the split between the validation and train sets is done by the constructor's of the dataset, even if you can fine thune it.
If you want to use an homemade dataset or a homemade model, you will have to use the [dataset class.](#dataset-generation)
Note that this parameter is not mandatory as the MNIST dataset is selected by default. 

### Set some ML parameters

Even if default training values are provided, it is strongly advised to adapt these to your particular use case. 
For instance you might want your training to go for 10 epochs and 3 minibatches per epoch.

```python
from mplc.scenario import Scenario
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

Here is an example of an accuracy plot for 3 partners

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

To use contributivity measurement tools, you will have to change the parameters of your `Scenario` object

```python
from mplc.scenario import Scenario
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

`dataset`: `None` (default), `datasets.Dataset object`. If None, the dataset provided by the `dataset_name` will be used.

`dataset_name`: `'mnist'` (default), `'cifar10'`, `'esc50'`, `'imdb'` or `'titanic'`
MNIST, CIFAR10, ESC50, IMDB and Titanic are currently supported. They come with their associated modules in `/datasets` for loading data, pre-processing inputs, and define a model architecture.\
For each dataset, it is possible to provide a path to model weights learned from a previous coalition. Use `'random_initialization'` if you want a random initialization or an empty value as in one of the two following syntaxes:
You can also use your own dataset, with the class Dataset. The [Tutorial 2](https://github.com/SubstraFoundation/distributed-learning-contributivity/blob/master/notebooks/examples/2%20_Sentiment140.ipynb) provides more explicit information.

**Note on validation and test datasets**:

- The dataset modules must provide separated train and test sets (referred to as global train set and global test set).
- The global train set is then further split into a global train set and a global validation set.
In the multi-partner learning computations, the global validation set is used for early stopping and the global test set is used for performance evaluation.
- The global train set is split amongst partner (according to the scenario configuration) to populate the partner's local datasets.
- For each partner, the local dataset can be split into separated train, validation and test sets, depending on the dataset configuration. Currently, the local validation and test set are not used, but they are available for further developments of multi-partner learning and contributivity measurement approaches.

`dataset_proportion`: `float` (default: `1`). 
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

<a id="sample_split_option"></a>
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

![Example of the advanced split option](../../img/advanced_split_example.png)

`corrupted_datasets`: `[not_corrupted (default), shuffled or corrupted]`  
Enables to artificially corrupt the data of one or several partners:

- `not_corrupted`: data are not corrupted
- `shuffled`: labels are shuffled randomly, not corresponding anymore with inputs
- `corrupted`: labels are all offseted of `1` class

Example: `[not_corrupted, not_corrupted, not_corrupted, shuffled]`

### Configuration of the collaborative and distributed learning

There are several parameters influencing how the collaborative and distributed learning is done over the datasets of the partners. The following schema introduces certain definitions used in the below description of parameters:

![Schema epochs mini-batches gradient updates](../../img/epoch_minibatch_gradientupdates.png)

`multi_partner_learning_approach`: `'fedavg'` (default), `'seq-pure'`, `'seq-with-final-agg'` or `'seqavg'`  
Define the multi-partner learning approach, among the following as described by the schemas:

- `'fedavg'`: stands for federated averaging

    ![Schema fedavg](../../img/collaborative_rounds_fedavg.png)

- `'seq-...'`: stands for sequential and comes with 2 variations, `'seq-pure'` with no aggregation at all, and `'seq-with-final-agg'` where an aggregation is performed before evaluating on the validation set and test set (on last mini-batch of each epoch) for mitigating impact when the very last subset on which the model is trained is of low quality, or corrupted, or just detrimental to the model performance.

    ![Schema seq](../../img/collaborative_rounds_seq.png)

- `'seqavg'`: stands for sequential averaging

    ![Schema seqavg](../../img/collaborative_rounds_seqavg.png)

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

```sh
- "Shapley values"
- "Independent scores"
- "TMCS"
- "ITMCS"
- "IS_lin_S"
- "IS_reg_S"
- "AIS_Kriging_S"
- "SMCS"
- "WR_SMC"
- "Federated SBS linear"
- "Federated SBS quadratic"
- "Federated SBS constant"
- "LFlip"
```

The methods are detailed below: 
- **Independent training**:

  - `["Independent scores"]` **Performance scores** of models trained independently on each partner

- [**Shapley values**](https://arxiv.org/pdf/1902.10275.pdf):  

  These indicators seem to be very good candidates to measure the contributivity of each data providers, because they are usually used in game theory to fairly attributes the gain of a coalition game amongst its players, which is exactly what we are looking for here.

  A coalition game is a game where players form coalitions and each coalitions gets a score according to some rules. The winners are the players who manage to be in the coalition with the best score. Here we can consider each data provider is a player, and that forming a coalition is building a federated model using the dataset of each player within the coalition. The score of a coalition is then the performance on a test set of the federated model built by the coalition.

  To attributes a part of the global score to each player/data providers, we can use the Shapley values. To define the Shapley value we first have to define the "increment" in performance of a player in a coalition. Such "increment" is the performance of the coalition minus the performance of the coalition without this player. The Shapley value of a player is a properly weighted average of its "increments" in every possible coalition.

  The computation of the Shapley Values quickly becomes intensive when the number of players increases. Indeed to compute the increment of a coalition, we need to fit two federated model, and we need to do this for every possible coalitions. If *N* is the number of players we have to do *2^N* fits to compute the Shapley values of each players. As this is quickly too costly, we are considering estimating the Shapley values rather then computing it exactly. The estimation methods considered are:

  - `["Shapley values"]` **The exact Shapley Values computation**:  

  Given the limited number of data partners we consider at that stage it is possible to actually compute the Shapley Values with a reasonable amount of resources.

  - **[Monte-Carlo Shapley](https://arxiv.org/pdf/1902.10275.pdf) approximation** (also called permutation sampling):  
  As the Shapley value is an average we can estimate it using the Monte-Carlo method. Here it consists in sampling a reasonable number of increments (says a hundred per player) and to take the average of the sampled increments of a player as the estimation of the Shapley value of that player.

  - `["TMCS"]` **[Truncated Monte-Carlo Shapley](https://arxiv.org/pdf/1904.02868.pdf) approximation**:  
  The idea of Truncated Monte-Carlo is that, for a large coalition, the increments of a player are usually small, therefore we can consider their value is null instead of spending computational power to compute it. This reduce the number of times we have to fit a model, but adds a small bias in the estimation.

  - `["ITMCS"]` **Interpolated Truncated Monte-Carlo Shapley**:  

  This method is an attempt to reduce the bias of the Truncated Monte-Carlo Shapley method. Here we do not consider that the value of an increment of a large coalition is null, but we do a linear interpolation to better approximate its value.

- **Importance sampling methods**:

  Importance sampling is a method to reduce the number of sampled increments in the Monte-Carlo method while keeping the same accuracy. It consists in sampling the increments according to non-uniform distribution, giving more chance for big increment than for small increment to be sampled. The bias induced by altering the sampling distribution is canceled by properly weighting each sample: if an increment is sampled with *X* times more chances, then we weight it by *1/X*. Note that this require to know the value of increment before computing them, so in practice we try to guess the value of the increment. We inflate, resp. deflate, the probability of sampling an increment if we guess that its value is high, resp. small. We designed three ways to guess the value of increments, which lead to three different importance sampling methods:

  - `["IS_lin_S"]` **Linear importance sampling**
  - `["IS_reg_S"]` **Regression importance sampling**
  - `["AIS_Kriging_S"]` **Adaptive Kriging importance sampling**

- **[Stratified Monte Carlo Shapley](https://arxiv.org/pdf/1904.02868.pdf)**:

  "Stratification and with proper allocation" is another method to reduce the number of sampled increments in the Monte-Carlo method while keeping the same accuracy. There are two ideas behind this method:

  1. The Shapley value is a mean of means taken on strata of increments. A strata of increments corresponds to all the increments of coalitions with the same number of players. We can estimate the means on each strata independently rather than the whole mean, this improves the accuracy and reduces the number of increments to sample.
  1. We can allocate a different amount of sampled increment to each mean of a strata. If we allocate more sample to the stratas where the increments value varies more, we can reduce the accuracy even more.

  As we can estimate the mean of a strata by sampling with replacement of without replacement, it gives two approximation methods:

  - `["SMCS"]` **Stratified Monte Carlo Shapley with replacement**
  - `["WR_SMC"]` **Stratified Monte Carlo Shapley without replacement**

- **Partner Valuation by Reinforcement Learning**:

    With PVRL, we modify the learning process of the main model so it includes a dataset's partner valuation part. Namely we assign weight to each dataset, and at each learning step these weights are used to sample the learning batch. These weight are updated at each learning iteration of the main model using the REINFORCE method.

   - `["PVRL"]` **Partner Valuation by Reinforcement Learning** 
    
- **Federated step-by-step**:

    Federated step by step contributivity methods measure the performance variation on the global validation dataset after each minibatch training - These methods give an estimation on how the model improved on every node.
    The methods are best suited for federated averaging learning.
    For each computation round, the contributivity for each partner is calculated as the ratio between the validation score of the newly trained model for each partner and the validation score from the previously trained collective model.
    Initial rounds (10%) and final rounds (10%) are discarded from calculation as performance increments from the first minibatches might be huge and increments form the last minibatches might be very noisy. Discarded proportions are for now set in the code.

    3 contributivity methods are proposed to adjust the importance of last computation rounds compared to the first ones:
    - `["Federated SBS linear"]` - Linear importance increase between computation rounds (1000th round weights 1000 times first round)
    - `["Federated SBS quadratic"]` - Quadratic importance increase between computation rounds (1000th round weights 10e6 times first round)
    - `["Federated SBS constant"]`- Constant importance increase between computation rounds (1000th round weights same as first round)
    
- [In progress] **Label Flipping**
    
    Label Flipping method provides a way to detect mislabelled datasets.
    The main idea is, while training the model, to learn the probability of a label to be flipped in another, inside each partner dataset. 
    Then, we flip the label of the noisy data to the most likely right label and train the main model on these likely right data.  
    A contributivity measure can be inferred from partner's matrices of flip-probability, by computing the exponential inverse of the Frobenius distance to the identity. 
    However this measure is to be handle carefully, the method is not designed specifically for contributivity measurement, but for mislabelled dataset detection. 
    
    - `["LFip]` - Label flipping method

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
This function provides the model which will be trained by the scenario object. Currently the library handles compiled Keras' model (see MNIST, ESC50, IMDB and CIFAR10 datasets), and Scikit-Learn Linear Regression (see the TITANIC dataset).  

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
    # (...)
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

*Work in progress*
