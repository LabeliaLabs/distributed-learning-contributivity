[![Build Status](https://travis-ci.org/SubstraFoundation/distributed-learning-contributivity.svg?branch=master)](https://travis-ci.org/SubstraFoundation/distributed-learning-contributivity)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SubstraFoundation/distributed-learning-contributivity/blob/master/run_experiment_on_google_collab.ipynb)
[![Discuss on Slack](https://img.shields.io/badge/chat-on%20slack-orange)](https://substra.us18.list-manage.com/track/click?e=2effed55c9&id=fa49875322&u=385fa3f9736ea94a1fcca969f)


# Exploration of dataset contributivity to a model in collaborative ML projects

## Introduction

In collaborative data science projects partners sometimes need to train a model on multiple datasets, contributed by different data providing partners. In such cases the partners might have to measure how much each dataset involved contributed to the performance of the model. This is useful for example as a basis to agree on how to share the reward of the ML challenge or the future revenues derived from the predictive model, or to detect possible corrupted datasets or partners not playing by the rules. We explore this question and the opportunity to implement some mechanisms helping partners in such scenarios to measure each dataset's *contributivity* (as *contribution to the performance of the model*).

### Context of this work

This work is being carried out in the context of collaborative research projects. It is work in progress. We would like to share it with various interested parties, research and business partners to get their feedback and potential contributions. This is why it is shared as open source content on Substra Foundation’s repositories.

### How to interact with this?

It depends in what capacity you are interested! For example:

- If you'd like to experiment right now by yourself multi-partner learning approaches and contributivity measurement methods, jump to section **[Using the code files](#using-the-code-files)**
- If you'd like to get in touch with the workgroup, jump to section **[Contacts, contributions, collaborations](#contacts-contributions-collaborations)**. If you are a student or a teacher, we love discussing student projects!
- If you are very familiar with this type of projects, well you can either have a look at section **[Ongoing work and improvement plan](#ongoing-work-and-improvement-plan)** or head towards [issues](https://github.com/SubstraFoundation/distributed-learning-contributivity/issues) and [PRs](https://github.com/SubstraFoundation/distributed-learning-contributivity/pulls) to see what's going on these days. We use the `help wanted` tag to flag issues on which help is particularly wanted, but other open issues would also very much welcome contributions

Should you have any question, [reach out](#contacts-contributions-collaborations) and we'll be happy to discuss how we could help.

## About this repository

In this repository, we benchmark different contributivity measurement approaches on a public dataset artificially split in a number of individual datasets, to mock a collaborative ML project.

The objective is to compare the contributivity figures obtained with the different approaches, and try to see how potential differences could be interpreted.

### Experimental approach

We want to start experimenting contributivity evaluations in collaborative data science and distributed multi-partner learning scenarios. Our exploration of this topic is in progress, as is this library and associated experiments. To make the most out of it, it is key to capitalize on this effort and develop it as a reproducible pipeline that can be improved, enriched, complemented over time.

For a start we made the following choices:

- What we want to compare (with the Shapley values being the baseline, see section below):
  - Contributivity relative values
  - Computation time
- Public datasets for experiments currently supported: MNIST, CIFAR10

### Structure of the library

This library can be broken down into 3 blocks:

1. Scenarios
1. Multi-partner learning approaches
1. Contributivity measurement approaches

#### Scenarios

From the start of this work it seemed very useful to be able to simulate different multi-partner settings to be able to experiment on them. For that, the library enables to configure scenarios by specifying the number of partners, what are the relative volume of data they have, how the data are distributed among them, etc. (see below section [Definition of collaborative scenarios](#definition-of-collaborative-scenarios) for all available parameters).

#### Multi-partner learning approaches

Once a given scenario configured, it seemed useful to choose how the multi-partner learning would be done. So far, 3 different approaches are implemented (federated averaging, sequential learning, sequential averaging). See below section [Configuration of the collaborative and distributed learning](#configuration-of-the-collaborative-and-distributed-learning) for descriptive schemas and additional ML-related parameters.

#### Contributivity measurement approaches

Finally, with given scenarios and multi-partner learning approaches, we can address contributivity measurement approaches. See below sections [Configuration of contributivity measurement methods to be tested](#configuration-of-contributivity-measurement-methods-to-be-tested) and [Contributivity measurement approaches studied and implemented](#contributivity-measurement-approaches-studied-and-implemented). 
  
### Using the code files

1. Define your mock scenario(s) in `config.yml` by changing the values of the suggested parameters of the 2 example scenarios (you can browse more available parameters in section [Config file parameters](#config-file-parameters) below). For example:

    ```yaml
    experiment_name: my_custom_experiment
    n_repeats: 5
    scenario_params_list:
     - dataset_name:
         - 'mnist'
         - 'cifar10'
       partners_count: 
         - 3
       amounts_per_partner: 
         - [0.4, 0.3, 0.3]
       samples_split_option: 
         - ['advanced', [[7, 'shared'], [6, 'shared'], [2, 'specific']]]
       multi_partner_learning_approach:
         - 'fedavg'
       aggregation_weighting: 
         - 'data_volume' 
         - 'uniform'
       epoch_count: 
         - 38
       methods:
         - ["Shapley values", "Independent scores", "TMCS"]
       minibatch_count: 
         - 20
       gradient_updates_per_pass_count:
         - 8
       proportion_dataset:
	 - 1
     - dataset_name:
         - 'mnist'
       partners_count: 
         - 2
       amounts_per_partner: 
         - [0.5, 0.5]
       samples_split_option: 
         - ['basic', 'random']
         - ['basic', 'stratified']
       multi_partner_learning_approach:
         - 'fedavg'
       aggregation_weighting: 
         - 'data_volume' 
         - 'uniform'
       epoch_count: 
         - 38
       methods:
         - ["Shapley values", "SMCS", "IS_lin_S", "IS_reg_S"]
       minibatch_count: 
         - 20
       gradient_updates_per_pass_count:
         - 8
       proportion_dataset:
	 - 1
    ```

   Under `scenario_params_list`, enter a list of sets of scenario(s). Each set starts with ` - dataset_name:` and must have only one `partners_count` value. The length of `amount_per_partners`, `corrupted_datasets` (and `samples_split_option` when the advanced definition is used) must match the `partner_counts` value. If for a given parameter multiple values are specified, e.g. like for `agregation_weighting` in the first scenario set of the above example, all possible combinations of parameters will be assembled as separate scenarios and run.
   
1. Then execute `main.py -f config.yml`. Add the `-v` argument if you want a more verbose output. 

1. A `results.csv` file will be generated in a new folder for your experiment under `/experiments/<your_experiment>`. You can read this raw `results.csv` file or use the notebooks in `/notebooks`.  

   **Note**: example experiment(s) are stored in folder `/saved_experiments` to illustrate the use of the library. The notebooks include graphs, like for example the following:  
   ![Example graphs](./img/results_graphs_example.png)

### Config file parameters

#### Experiment-level parameters

An experiment regroups one or several scenarios to be run.

`experiment_name`: `str`  
How the experiment will be named in output files.  
Example: `experiment_name: my_first_experiment`
  
`n_repeats`: `int`  
How many times the same experiment is run.  
Example: `n_repeats: 2`

#### Scenario-level parameters

##### Choice of dataset

`dataset_name`: `'mnist'` (default) or `'cifar10'`  
MNIST and CIFAR10 are currently supported. They come with their associated modules in `/datasets` for loading data, pre-processing inputs, and define a model architecture.

**Note on validation and test datasets**:  
- The dataset modules must provide separated train and test sets (referred to as global train set and global test set).
- The global train set is then further split into a global train set and a global validation set.
In the multi-partner learning computations, the global validation set is used for early stopping and the global test set is used for performance evaluation.
- The global train set is split amongst partner (according to the scenario configuration) to populate the partner's local datasets.
- For each partner, the local dataset is split into separated train, validation and test sets. Currently, the local validation and test set are not used, but they are available for further developments of multi-partner learning and contributivity measurement approaches.
- 

`proportion_dataset`: `float` (default: `1`)
This argument allows you to make computation on a sub-dataset of the provided datasets.
This is the proportion of the dataset (initialy the train and test sets) which is randomly selected to create a sub-dataset,
it's done before the creation of the global validation set. 
You have to ensure that `0 < proportion_dataset <= 1`


##### Definition of collaborative scenarios

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

![Example of the advanced split option](img/advanced_split_example.png)

`corrupted_datasets`: `[not_corrupted (default), shuffled or corrupted]`  
Enables to artificially corrupt the data of one or several partners:

- `not_corrupted`: data are not corrupted
- `shuffled`: labels are shuffled randomly, not corresponding anymore with inputs
- `corrupted`: labels are all offseted of `1` class
  
Example: `[not_corrupted, not_corrupted, not_corrupted, shuffled]`

##### Configuration of the collaborative and distributed learning 

There are several parameters influencing how the collaborative and distributed learning is done over the datasets of the partners. The following schema introduces certain definitions used in the below description of parameters:

![Schema epochs mini-batches gradient updates](img/epoch_minibatch_gradientupdates.png)

`multi_partner_learning_approach`: `'fedavg'` (default), `'seq-pure'`, `'seq-with-final-agg'` or `'seqavg'`  
Define the multi-partner learning approach, among the following as described by the schemas:

- `'fedavg'`: stands for federated averaging
    
    ![Schema fedavg](img/collaborative_rounds_fedavg.png)
  
- `'seq-...'`: stands for sequential and comes with 2 variations, `'seq-pure'` with no aggregation at all, and `'seq-with-final-agg'` where an aggregation is performed before evaluating on the validation set and test set (on last mini-batch of each epoch) for mitigating impact when the very last subset on which the model is trained is of low quality, or corrupted, or just detrimental to the model performance.
    
    ![Schema seq](img/collaborative_rounds_seq.png)
  
- `'seqavg'`: stands for sequential averaging
    
    ![Schema seqavg](img/collaborative_rounds_seqavg.png)
    
Example: `multi_partner_learning_approach: 'seqavg'`

`aggregation_weighting`: `'uniform'` (default), `'data_volume'` or `'local_score'`  
After a training iteration over a given mini-batch, how individual models of each partner are aggregated:

- `'uniform'`: simple average (non-weighted)
- `'data_volume'`: average weighted with per the amounts of data of partners (number of data samples)
- `'local_score'`: average weighted with the performance (on a central validation set) of the individual models

Example: `aggregation_weighting: 'data_volume'`

`epoch_count`: `int` (default: `40`)  
Number of epochs, i.e. of passes over the entire datasets. Superseded when `is_early_stopping` is set to `true`.  
Example: `epoch_count: 30`

`minibatch_count`: `int` (default: `20`)  
Within an epoch, the learning on each partner's dataset is not done in one single pass. The partners' datasets are split into multiple *mini-batches*, over which learning iterations are performed. These iterations are repeated for all *mini-batches* into which the partner's datasets are split at the beginning of each epoch. This gives a total of `epoch_count * minibatch_count` learning iterations.  
Example: `minibatch_count: 20`

`gradient_updates_per_pass_count`: `int` (default: `8`)  
The ML training implemented relies on Keras' `.fit()` function, which takes as argument a `batch_size` interpreted by `fit()` as the number of samples per gradient update. Depending on the number of samples in the train dataset, this defines how many gradient updates are done by `.fit()`. The `gradient_updates_per_pass_count` parameter enables to specify this number of gradient updates per `.fit()` iteration (both in multi-partner setting where there is 1 `.fit()` iteration per mini-batch, and in single-partner setting where there is 1 `.fit()` iteration per epoch).  
Example: `gradient_updates_per_pass_count: 5`

`is_early_stopping`: `True` (default) or `False`  
When set to `True`, the training phases (whether multi-partner of single-partner) are stopped when the performance on the validation set reaches a plateau.  
Example: `is_early_stopping: False` 

**Note:** to only launch the distributed learning on the scenarios (and no contributivity measurement methods), simply omit the `methods` parameter (see section [Configuration of contributivity measurement methods to be tested](#configuration-of-contributivity-measurement-methods-to-be-tested) below).

##### Configuration of contributivity measurement methods to be tested

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

##### Miscellaneous

`is_quick_demo`: `True` or `False` (default)  
When set to `True`, the amount of data samples and the number of epochs and mini-batches are significantly reduced, to minimize the duration of the run. This is particularly useful for quick demos or debugging.  
Example: `is_quick_demo: True`

### Contributivity measurement approaches studied and implemented

- [done, short name "Independent scores"] **Performance scores** of models trained independently on each partner

- [**Shapley values**](https://arxiv.org/pdf/1902.10275.pdf):  

  These indicators seem to be very good candidates to measure the contributivity of each data providers, because they are usually used in game theory to fairly attributes the gain of a coalition game amongst its players, which is exactly what we are looking for here.
  
  A coalition game is a game where players form coalitions and each coalitions gets a score according to some rules. The winners are the players who manage to be in the coalition with the best score. Here we can consider each data provider is a player, and that forming a coalition is building a federated model using the dataset of each player within the coalition. The score of a coalition is then the performance on a test set of the federated model built by the coalition.

  To attributes a part of the global score to each player/data providers, we can use the Shapley values. To define the Shapley value we first have to define the "increment" in performance of a player in a coalition. Such "increment" is the performance of the coalition minus the performance of the coalition without this player. The Shapley value of a player is a properly weighted average of its "increments" in every possible coalition.
  
  The computation of the Shapley Values quickly becomes intensive when the number of players increases. Indeed to compute the increment of a coalition, we need to fit two federated model, and we need to do this for every possible coalitions. If *N* is the number of players we have to do *2^N* fits to compute the Shapley values of each players. As this is quickly too costly, we are considering estimating the Shapley values rather then computing it exactly. The estimation methods considered are:

  - [done, short name "Shapley values"] **The exact Shapley Values computation**:  
  Given the limited number of data partners we consider at that stage it is possible to actually compute the Shapley Values with a reasonable amount of resources.
    
  - **[Monte-Carlo Shapley](https://arxiv.org/pdf/1902.10275.pdf) approximation** (also called permutation sampling):  
  As the Shapley value is an average we can estimate it using the Monte-Carlo method. Here it consists in sampling a reasonable number of increments (says a hundred per player) and to take the average of the sampled increments of a player as the estimation of the Shapley value of that player.
    
  - [done, short name "TMCS"] **[Truncated Monte-Carlo Shapley](https://arxiv.org/pdf/1904.02868.pdf) approximation**:  
  The idea of Truncated Monte-Carlo is that, for a large coalition, the increments of a player are usually small, therefore we can consider their value is null instead of spending computational power to compute it. This reduce the number of times we have to fit a model, but adds a small bias in the estimation.
    
  - [done, short name "ITMCS"] **Interpolated Truncated Monte-Carlo Shapley**:  
  This method is an attempt to reduce the bias of the Truncated Monte-Carlo Shapley method. Here we do not consider that the value of an increment of a large coalition is null, but we do a linear interpolation to better approximate its value.
    
- **Importance sampling methods**:

  Importance sampling is a method to reduce the number of sampled increments in the Monte-Carlo method while keeping the same accuracy. It consists in sampling the increments according to non-uniform distribution, giving more chance for big increment than for small increment to be sampled. The bias induced by altering the sampling distribution is canceled by properly weighting each sample: if an increment is sampled with *X* times more chances, then we weight it by *1/X*. Note that this require to know the value of increment before computing them, so in practice we try to guess the value of the increment. We inflate, resp. deflate, the probability of sampling an increment if we guess that its value is high, resp. small. We designed three ways to guess the value of increments, which lead to three different importance sampling methods:
  
  - [done, short name "IS_lin_S"] **Linear importance sampling**
  - [done, short name "IS_reg_S"] **Regression importance sampling**
  - [done, short name "AIS_Kriging_S"] **Adaptative Kriging importance sampling**

- **[Stratified Monte Carlo Shapley](https://arxiv.org/pdf/1904.02868.pdf)**:

  "Stratification and with proper allocation" is another method to reduce the number of sampled increments in the Monte-Carlo method while keeping the same accuracy. There are two ideas behind this method:

  1. The Shapley value is a mean of means taken on strata of increments. A strata of increments corresponds to all the increments of coalitions with the same number of players. We can estimate the means on each strata independently rather than the whole mean, this improves the accuracy and reduces the number of increments to sample.
  1. We can allocate a different amount of sampled increment to each mean of a strata. If we allocate more sample to the stratas where the increments value varies more, we can reduce the accuracy even more.
   
  As we can estimate the mean of a strata by sampling with replacement of without replacement, it gives two approximation methods:
  
  - [done, short name "SMCS"] **Stratified Monte Carlo Shapley with replacement**
  - [done, short name "WR_SMC"] **Stratified Monte Carlo Shapley without replacement**
    
- [future prospect] [**Data Valuation by Reinforcement Learning**](https://arxiv.org/pdf/1909.11671.pdf) (DVRL):

  With DVRL, we modify the learning process of the main model so it includes a data valuation part. Namely we use a small neural network to assign weight to each data, and at each learning step these weights are used to sample the learning batch. These weight are updated at each learning iteration of the main model using the REINFORCE method.
  
- [future prospect] **Federated step-by-step increments**:

  See open [issue #105](https://github.com/SubstraFoundation/distributed-learning-contributivity/issues/105).
  
### Ongoing work and improvement plan

The current work focuses on the following 4 priorities:

1. Improve the **[multi-partner learning approaches](https://github.com/SubstraFoundation/distributed-learning-contributivity/projects/4)**
1. Continue developing new **[contributivity measurement methods](https://github.com/SubstraFoundation/distributed-learning-contributivity/projects/3)**
1. Perform **[experiments](https://github.com/SubstraFoundation/distributed-learning-contributivity/projects/1)** and gain experience about best-suited contributivity measurement methods in different situations
1. Make the library **[agnostic/compatible with other datasets and model architectures](https://github.com/SubstraFoundation/distributed-learning-contributivity/projects/2)**

There is also a transverse, continuous improvement effort on **[code quality, readibility, optimization](https://github.com/SubstraFoundation/distributed-learning-contributivity/projects/5)**.

This work is collaborative, enthusiasts are welcome to comment open issues and PRs or open new ones.

## Contacts, contributions, collaborations

Should you be interested in this open effort and would like to share any question, suggestion or input, you can use the following channels:
  - This Github repository (issues or PRs)
  - Substra Foundation's [Slack workspace](https://substra-workspace.slack.com/join/shared_invite/zt-cpyedcab-FHYgpy08efKJ2FCadE2yCA), channel `#workgroup-mpl-contributivity`
  - Email: hello@substra.org
  - Come meet with us at La Paillasse (Paris, France), Le Palace (Nantes, France) or Studio Iconosquare (Limoges, France)
  
 ![logo Substra Foundation](./img/substra_logo_couleur_rvb_w150px.png)
