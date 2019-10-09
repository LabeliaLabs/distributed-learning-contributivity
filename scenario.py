# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:43:26 2019

This enables to parameterize a desired scenario of data splitting amond nodes.
The scenario is then processed by data_splitting.py

@author: @bowni
"""

from keras.datasets import mnist
import json
import numpy as np


class Scenario:
  def __init__(self):
      
    self.dataset_name = 'MNIST'
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Truncate dataset for quicker debugging/testing
    self.x_train = x_train[:6000]
    self.y_train = y_train[:6000]
 
    self.x_test = x_test[:3000]
    self.y_test = y_test[:3000]
    
    # Define the desired number of independant nodes
    # Nodes mock different partners in a collaborative data science project
    self.nodes_count = 3

    # Configure the desired respective datasets sizes of the nodes
    # Should the nodes receive an equivalent amount of samples each...
    # ... or receive different amounts?
    # Define the percentages of samples per node
    # Sum has to equal 1 and number of items has to equal NODES_COUNT
    self.amounts_per_node = [0.2, 0.3, 0.5]
    
    # Configure if nodes get overlapping or distinct samples
    # Should the nodes receive data from distinct categories...
    # ... or just random samples?
    self.overlap_or_distinct = 'Distinct' # Toggle between 'Overlap' and 'Distinct'
    
    # Define if test data should be split between nodes...
    # ... or if each node should refer to the complete test set
    self.testset_option = 'Global' # Toggle between 'Global' and 'Split'
    
    # TODO
    self.nodes_list = []
    self.shapley_values = []
    self.computation_time = 0


  def to_json(self):
            
    # Omit np.ndarray when converting to JSON.
    # TODO : find better way to not outputs training/testing data to JSON
    def default(o):
        if isinstance(o, np.ndarray):
            return np.nan
    
    return json.dumps(self.__dict__, default=default)
        

  def to_file(self):
    
    json_data = self.to_json()
    # TODO create results folder and better filename
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4, sort_keys=True)

    
