[![Build Status](https://travis-ci.org/SubstraFoundation/distributed-learning-contributivity.svg?branch=master)](https://travis-ci.org/SubstraFoundation/distributed-learning-contributivity)

# Exploration of dataset contributivity to a model in collaborative ML projects

## Introduction

In data science projects involving multiple data providers, each one contributing data for the training of the same model, the partners might have to agree on how to share the reward of the ML challenge or the future revenues derived from the predictive model. We explore this question and the opportunity to implement some mechanisms helping partners to easily agree on a value sharing model.

## Context of this work

This work is being carried out in the context of the [HealthChain research consortium](https://www.substra.ai/en/healthchain-project). It is work in progress, early stage. We would like to share it with various interested parties and business partners to get their feedback and potential contributions. This is why it is shared as open source content on Substra Foundation’s repositories.

## Exploratory document

An exploratory document provides a deeper context description and details certain contributivity measurement approaches. This document can be found [here](https://docs.google.com/document/d/1dILvplN7h3-KB6OcHFNx9lSpAKyaBrwNaIRQ9j6XDT8/edit?usp=sharing). It is an ongoing effort, eager to welcome collaborations, feedbacks and questions.

## About this repository

In this repository, we benchmark different contributivity measurement approaches on a public dataset artificially split in a number of individual datasets, to mock a collaborative ML project.

The objective is to compare the contributivity figures obtained with the different approaches, and try to see how potential differences could be interpreted.

### Experimental approach

We want to start experimenting contributivity evaluations in collaborative data science / distributed learning scenarios. At this stage this cannot be a thorough and complete experimentation though, as our exploration of the topic is in progress. To make the most out of it, it is key to capitalize on this effort and develop it as a reproducible pipeline that we will be able to improve, enrich, complement over time.

* Public dataset of choice: MNIST
* Collaborative data science scenarios - Parameters:
    * Overlap of respective datasets: distinct (by stratifying MNIST figures) vs. overlapping (with a randomized split)
    * Size of respective datasets: equivalent vs. different
    * Number of data partners: 3 databases A, B, C is our default scenario, but this is to be parameterized
* ML algorithm: CNN adapted to MNIST, not too deep so it can run on CPU
* Distributed learning approach: federated learning (other approaches to be tested in future improvements of this experiment)
* Contributivity evaluation approach:
    * [done] Performance scores of models trained independently on each node
    * [futur prospect] [Data Valuation by Reinforcement Learning](https://arxiv.org/pdf/1909.11671.pdf) (DVRL)
     With DVRL, we modifidy the learning process of the main model so it includes a data valuation part. Namely we use a small neural network to assign weight to each data, and at each learning step these weights are used to sample the learning batch.  These weight are updated at each learning iteration of the main model using the REINFORCE method.   
    * [done] [Shapley values](https://arxiv.org/pdf/1902.10275.pdf) :
     These indicators seem to be very good candidates to measure the contributivity of each data providers, because they are usually used in game theory to fairly attributes the gain of a coalitional game amongst its players, which is exactly want we are looking for here.<br/><br/>
A coalition game is a game where players form coalitions and each coalitions gets a score according to some rules. The winners are the players who manage to be in the coalition with the best score. Here we can consider each data provider is a player, and that forming a coalition is building a federated model using the dataset of each player within the coalition. The score of a coalition is then the performance on a test set of the federated model built by the coalition.<br/><br/>
To attributes a part of the global score to each player/data providers, we can use the Shapley values. To define the Shapley value we first have to define the "increment" in performance of a player in a coalition. Such "increment" is the performance of the coalition minus the performance of the coalition without this player. The Shapley value of a player is a properly weighted average of its "increments" in every possible coalition. <br/><br/> 
The computation of the Shapley Values quickly becomes intensive when the number of players increases. Indeed to compute the increment of a coalition, we need to fit two federated model, and we need to do this for every possible coalitions. If *N* is the number of players we have to do *2^N* fits to compute the Shapley values of each players. As this is quickly too costly, we are considering estimating the Shapley values rather then computing it exactly. The estimation methods considered are:
        * [done] The exact Shapley Values computation
        Given the limited number of data partners we consider at that stage it is possible to actually compute the Shapley Values with a reasonable amount of resources. 
        * [done] [Monte-Carlo Shapley](https://arxiv.org/pdf/1902.10275.pdf) approximation (also called permutation sampling)
        As the sahpley value is an average we can estimate it using the Monte-Carlo method. Here it consists in sampling a reasonable number of increments (says a hundred per player) and to take the average of the sampled increments of a player as the estimation of the Shapley value of that player.
        * [done] [Truncated Monte-Carlo Shapley](https://arxiv.org/pdf/1904.02868.pdf) approximation
        The idea of Truncated Monte-Carlo is that, for big coalition, the increments of a player are usually small, therefore we can consider their value is null instead of spending computional power to compute it. This reduce the number of times we have to fit a model, but adds a small bias the estimation.
        * [done] Interpolated truncated Monte-Carlo
        This method is an attempt to reduce the bias of the Truncated monte-Carlo method. Here we do not consider the value of an increment of a big coalition is null, but we do a linear interpolation to better approximate its value.
        * [done] Importance sampling methods
        Importance sampling is a method to reduce the number of sampled increments in the Monte-Carlo method while keeping the same accuracy. It consists in sampling the increments according to non-uniform distribution, giving more chance for big increment than for small increment to be sampled. The bias induced by altering the sampling distribution is canceled by properly weighting each sample: If an increment is sampled with *X* times more chances, then we weight it by *1/X*. Note that this require to know the value of increment before computing them, so in practice we try to guess the value of the increment. We inflate, resp. deflate, the probability of sampling an increment if we guess its value is big, resp. small. We designed three ways to guess the value of increments, which lead to three different importance sampling methods: 
            * [done] Linear importance sampling
            * [done] Regression importance sampling
            * [done] Adaptative kriging importance sampling
         * [done] [Stratified Monte Carlo Shapley](https://arxiv.org/pdf/1904.02868.pdf)
         "Stratification and with proper allocation" is another method to reduce the number of sampled increments in the Monte-Carlo method while keeping the same accuracy. There are two ideas behind this method:  1) the Sapley value is a mean of means taken on strata of increments. A strata of increments corresponds the all the increments of coalition with the same number of players. We can estimate the means on each stata independently rather than the whole mean, this improves the accuracy and reduces the number of increments to sample.  2) We can allocate a different amount of sampled increment to each mean of a strata. If we allocate more sample to the stratas where the increments value varies more, we can reduce the accuracy even more. As we can estimate the mean of a strata by sampling with replacement of without replacement, it gives two approximation methods:
            * [done] Stratified Monte Carlo Shapley with replacement
            * [done] Stratified Monte Carlo Shapley without replacement 
* Comparison variables (baseline: Shapley value)
    * Contributivity relative values
    * Computation time
  
### Using the code files

- Define your mock scenario(s) in `config.yml` by changing the values of the suggested parameters of the custom scenario (you can browse more available parameters in `scenario.py`). For example:
    ```yaml
    experiment_name: my_custom_experiment
    n_repeats: 10
    scenario_params_list:
     - nodes_counts: 3
       amounts_per_node: [0.4, 0.3, 0.3] 
       samples_split_option: 'Random'
       aggregation_weighting: 'data-volume'
       single_partner_test_mode: 'global'
       epoch_count: 38
       minibatch_count: 20
     - nodes_counts: 4
       amounts_per_node: [0.3, 0.3, 0.1, 0.3] 
       samples_split_option: 'Stratified'
       aggregation_weighting: 'data-volume'
       single_partner_test_mode: 'global'
       epoch_count: 38
       minibatch_count: 20
    ```
- Then execute `simulation_run.py -f config.yml`
- A `results.csv` file will be generated in a new folder for your experiment under `/experiments`. You can read this raw `results.csv` file or use the `analyse_results.ipynb` notebook to quickly generate figures.

## Contacts

Should you be interested in this open effort and would like to share any question, suggestion or input, you can use the following channels:
  - This Github repository (issues, PR...)
  - Substra Foundation's [Slack workspace](https://substra-workspace.slack.com/join/shared_invite/zt-cpyedcab-FHYgpy08efKJ2FCadE2yCA)
  - Email: hello@substra.org
  - Come meet with us at La Paillasse (Paris, France), Le Palace (Nantes, France) or Studio Iconosquare (Limoges, France)
