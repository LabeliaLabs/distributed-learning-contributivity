# Exploration of dataset contributivity to a model in collaborative ML projects

## Introduction

In data science projects involving multiple data providers, each one contributing data for the training of the same model, the partners might have to agree on how to share the reward of the ML challenge or the future revenues derived from the predictive model. We explore this question and the opportunity to implement some mechanisms helping partners to easily agree on a value sharing model.

## Context of this work

This work is being carried out in the context of the [HealthChain research consortium](https://www.substra.ai/en/healthchain-project). It is work in progress, early stage. We would like to share it with various interested parties and business partners to get their feedback and potential contributions. This is why it is shared as open source content on Substra Foundation’s repositories.

## Work-in-progress exploration document

The work-in-progress document describing this exploration can be found [here](https://docs.google.com/document/d/1dILvplN7h3-KB6OcHFNx9lSpAKyaBrwNaIRQ9j6XDT8/edit?usp=sharing). It is an ongoing effort, eager to welcome collaborations, feedbacks and questions.

## About this repository

In this repository, we would like to benchmark the different contributivity measurement approaches described in the document on a set of public datasets (or a single dataset artificially split in a number of individual datasets, to mock a collaborative ML project).

The objective is to compare the contributivity figures obtained with the different approaches, and try to see how potential differences could be interpreted.

### Experimental plan H2 2019

TODO: reformulate

We want to start experimenting contributivity evaluations in collaborative data science / distributed learning scenarios. At this stage this cannot be a thorough and complete experimentation though, as our exploration of the topic is in progress and we can dedicate only a limited amount of time and energy to this project. To make the most out of it, it is key to capitalize on this first experiment and develop it as a reproducible pipeline that we will be able to improve, enrich, complement over time.

- Public dataset of choice: MNIST
- Collaborative data science scenarios - Parameters:
  - Overlap of respective datasets: distinct (by stratifying MNIST figures) vs. overlapping (randomized split)
  - Size of respective datasets: equivalent vs. different
  - Number of data partners: 3 databases A, B, C (to be parameterized in future improvements of this experiment)
- ML algorithm: CNN, not too deep so it can run on CPU
- Distributed learning approach: federated learning (other approaches to be tested in future improvements of this experiment)
- Contributivity evaluation approach:
  - The approaches we would like to compare (further described in the document linked above):
    - [done] Shapley value: given the limited number of data partners we consider at that stage it is possible to actually 
compute it with a reasonable amount of resources
    - [done] Performance scores of models trained independently on each node
    - Truncated Monte Carlo Shapley approximation adapted for datasets instead of individual datum
    - Federated learning step-by-step
    - (More approaches to be tested in future improvements of this experiment)
  - Comparison variables (baseline: Shapley value)
    - Contributivity relative values
    - Computation time
  
### Using the code files

- Define your mock scenario(s) in `config.yml` by changing the values of the suggested parameters of the custom scenario (you can browse more available parameters in `scenario.py`)
- Clear `results.csv` if you want a clean sheet for starters
- Then execute `simulation_run.py -f config.yml`
- Consult results in the `results.csv` generated file or via the `analyse_results.ipynb` notebook

## Contacts

Should you be interested in this open effort and would like to share any question, suggestion or input, you can use the following channels:
  - This Github repository (issues, PR...)
  - Substra Foundation's [Slack workspace](https://substra-workspace.slack.com/join/shared_invite/zt-cpyedcab-FHYgpy08efKJ2FCadE2yCA)
  - Email: hello@substra.org
  - Come meet with us at La Paillasse (Paris, France), Le Palace (Nantes, France) or Studio Iconosquare (Limoges, France)

