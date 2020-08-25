# Distributed learning contributivity
# Work in progress

## Summary

- Quick start
- Results
- Contributivity mesuring methods

## Prerequisities

At root folder:
- `pip install -r requirements.txt`
- `pip install -i https://test.pypi.org/simple/ subtest==0.0.0.6`

Note: This is the temporary package for our library.

## Check out our tutorials!

For a better start and a quick understanding of how our library work, we recommand to take a look at our Notebook based tutorials.

## Quick start

### Important classes:

#### Dataset

This is where you define your dataset and relatives objects such as preprocessing functions and model generator.

##### Dataset

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
)
```

##### Model generator


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

##### Preprocessing
```python
def preprocess_dataset_labels(y):
    # Do stuff
    return y
```

#### Scenario

This is how you launch a collaborative round.

The steps to follow are:

- Define scenario parameters:

  There are 2 mandatory parametters for a collaborative run: `partners_count` and `amounts_per_partner`.

 Here is an example of how you should do it:
```
scenario_params = {
    'partners_count': 3,
    'amounts_per_partner': [0.2, 0.5, 0.3],
}
```
The first partner will have 20% of the total dataset, the second 50% and the third 30% for a total of 100%.

 If you don't want to use the entire dataset you can use the `dataset_proportion` parameter.

 You might also want to consider other parameters such as `epoch_count` or `minibatch_count`. See full params list at: [scenario-level-parameters](https://github.com/SubstraFoundation/distributed-learning-contributivity/blob/master/README.md#scenario-level-parameters)

- Create scenario:
```python
current_scenario = Scenario(scenario_params)
```

- Asignate your `dataset` object to the `current_scenario`:
```
current_scenario.dataset = dataset
```
- Split the validation and train sets:
```
current_scenario.dataset.train_val_split()
```
- Run the scenario:
```
run_scenario(current_scenario)
```

## Results

After a run, every information regarding the training phase will be available under the `multi_partner_learning` object in the `scenario.dataset.mpl` attribute

For instance: you can access `scenario.mpl.loss_collective_models`.

Here is a non exhaustive list of metrics available:

- `loss_collective_models`
- `score_matrix_collective_models`
- `score_matrix_per_partner`

Here is an example of an accuracy plot for 3 partner
```
import seaborn as sns
sns.set()

x= current_scenario.mpl.score_matrix_per_partner

x_collective = current_scenario.mpl.score_matrix_collective_models

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

Check out our Tutorial 3 for more informations.

## Contributivity mesuring methods

To use contributivity mesuring tools, you will have to set parameters to your `scenario` object.
```
scenario_params['methods'] = ["Shapley values"]
```

To access to the results use:
```
contributivity_score = current_scenario.contributivity_list
print(contributivity_score[0])
```

Check out our Tutorial 4 for more informations.


## Contacts, contributions, collaborations
Should you be interested in this open effort and would like to share any question, suggestion or input, you can use the following channels:

- This Github repository (issues or PRs)
- Substra Foundation's Slack workspace, channel #workgroup-mpl-contributivity
- Email: hello@substra.org
- Come meet with us at La Paillasse (Paris, France), Le Palace (Nantes, France) or Studio Iconosquare (Limoges, France)


# Work in progress
