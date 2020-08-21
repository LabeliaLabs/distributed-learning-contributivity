# Distributed learning contributivity

## Summary

## Prerequisities

At root folder :
- pip install -r requirements.txt
- pip install -i https://test.pypi.org/simple/ subtest==0.0.0.6

## Our tutorials

For a better start and a quick understanding of how our library work, we recommand to take a look at our Notebook based tutorials.

## Quick start

### Impmortant classes :

#### Dataset

This is where you define your dataset and relatives objects such as preprocessing functions and model generator.

##### Dataset

This is the structure of the Dataset generator :

dataset.Dataset(
    "name",
    X_train,
    X_test,
    y_train,
    y_test,
    input_shape,
    num_classes,
    preprocess_dataset_labels,
    generate_new_model_for_dataset
)

##### Model generator

##### Preprocessing


#### Scenario

This is how you launch a collaborative round.

The steps to follow are :

- define scenario params :
There are 2 mandatory parametters for a collaborative run : partners_count and amounts_per_partner.
Here is an example of how you should do it :

scenario_params = {
    'partners_count': 3,
    'amounts_per_partner': [0.2, 0.5, 0.3],
}

The first partner will have 20% of the total dataset, the second 50% and the third 30% for a total of 100%.

If you don't want to use the entire dataset you can use the dataset_proportion parameter.
