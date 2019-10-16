# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:43:26 2019

This enables to parameterize a desired scenario of data splitting amond nodes.
The scenario is then processed by data_splitting.py

@author: @bowni
"""

from keras.datasets import mnist
import os
import datetime


class Scenario:
  def __init__(self):
      
    self.dataset_name = 'MNIST'
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Add express option to limit the dataset size
    
    # Truncate dataset for quicker debugging/testing
    self.x_train = x_train
    self.y_train = y_train
 
    self.x_test = x_test
    self.y_test = y_test
    
    # Define the desired number of independant nodes
    # Nodes mock different partners in a collaborative data science project
    self.nodes_count = 3

    # Configure the desired respective datasets sizes of the nodes
    # Should the nodes receive an equivalent amount of samples each...
    # ... or receive different amounts?
    # Define the percentages of samples per node
    # Sum has to equal 1 and number of items has to equal NODES_COUNT
    self.amounts_per_node = [0.33, 0.33, 0.34]
    
    # Configure if nodes get overlapping or distinct samples
    # Should the nodes receive data from distinct categories...
    # ... or just random samples?
    # TODO find better names
    self.overlap_or_distinct = 'Overlap' # Toggle between 'Overlap' and 'Distinct'
    
    # Define if test data should be split between nodes...
    # ... or if each node should refer to the complete test set
    self.testset_option = 'Global' # Toggle between 'Global' and 'Split'
    
    # TODO
    self.nodes_list = []
    
    self.contributivity_list = []


  def append_contributivity(self, contributivity):
            
      self.contributivity_list.append(contributivity)
      

  def to_file(self):
    
    out = ''
    out += 'Dataset name: ' + self.dataset_name + '\n'
    out += 'Nodes_count: ' + str(self.nodes_count) + '\n'
    out += 'Amounts per node: ' + str(self.amounts_per_node) + '\n'
    out += 'Overlap or distinct: ' + self.overlap_or_distinct + '\n'
    out += 'Test set option: ' + self.testset_option + '\n'
    out += '\n'
    
    out += str(len(self.contributivity_list)) + ' contributivity methods: ' + '\n'

    for contrib in self.contributivity_list:
      out += str(contrib) + '\n\n'
       
    target_folder = 'results'
    os.makedirs(target_folder, exist_ok=True)
    
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%Hh%M")
    target_filename = now_str + '.txt'
    target_file_path = os.path.join(target_folder, target_filename)
    
    with open(target_file_path, 'w', encoding='utf-8') as f:
        f.write(out)