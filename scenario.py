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
import constants


class Scenario:
  def __init__(self, is_quick_demo=False):
      
    self.dataset_name = 'MNIST'
    (x_train, y_train), (x_test, y_test) = mnist.load_data()    
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
    
    # Configure if data samples are split between nodes randomly or in a stratified way...
    # ... so that they cover distinct areas of the samples space
    self.samples_split_option = 'Random' # Toggle between 'Random' and 'Stratified'
    
    # Define if test data should be distributed between nodes...
    # ... or if each node should refer to a centralised test set
    self.testset_option = 'Centralised' # Toggle between 'Centralised' and 'Distributed'
    
    # TODO
    self.nodes_list = []
    
    self.contributivity_list = []
    
    self.nb_epochs = constants.NB_EPOCHS

    if is_quick_demo:
        
        # Use les data and less epochs to speed up the computaions
        self.x_train = x_train[:600]
        self.y_train = y_train[:600]
        self.x_test = x_test[:100]
        self.y_test = y_test[:100]
        
        self.nb_epochs = 2
        
    
  def append_contributivity(self, contributivity):
            
      self.contributivity_list.append(contributivity)
      

  def to_file(self):
    
    out = ''
    out += 'Dataset name: ' + self.dataset_name + '\n'
    out += 'Number of data samples - train: ' + str(len(self.x_train)) + '\n'
    out += 'Number of data samples - test: ' + str(len(self.x_test)) + '\n'
    out += 'Nodes count: ' + str(self.nodes_count) + '\n'
    out += 'Percentages of data samples per node: ' + str(self.amounts_per_node) + '\n'
    out += 'Random or stratified split of data samples: ' + self.samples_split_option + '\n'
    out += 'Centralised or distributed test set: ' + self.testset_option + '\n'
    out += 'Number of epochs defined in learning algos: ' + str(self.nb_epochs) + '\n'
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