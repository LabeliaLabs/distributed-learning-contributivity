# -*- coding: utf-8 -*-
"""
This enables to parameterize a desired scenario of data splitting among nodes.
"""

from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import os
import datetime
import numpy as np
from node import Node
from pathlib import Path
import matplotlib.pyplot as plt
import uuid

class Scenario:
  def __init__(self, is_quick_demo=False):
      
    self.dataset_name = 'MNIST'
    (x_train, y_train), (x_test, y_test) = mnist.load_data()    
    # The train set has to be split into a train set and a validation set for early stopping (called 'esval' below)
    self.x_train, self.x_esval, self.y_train, self.y_esval = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
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
    
    self.federated_test_score = int
    
    self.node_list = []
    
    self.contributivity_list = []
    
    self.epoch_count = 40

    self.is_early_stopping = True
    
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%Hh%M")
    folder_name = now_str + '_' + uuid.uuid4().hex #TODO: this is quick hack so that the two disctincts scenario do not have the same name
    self.save_folder = Path('results') / Path(folder_name)
    os.makedirs(self.save_folder, exist_ok=True)

    if is_quick_demo:
        
        # Use less data and less epochs to speed up the computaions
        self.x_train = self.x_train[:1000]
        self.y_train = self.y_train[:1000]
        self.x_esval = self.x_esval[:100]
        self.y_esval = self.y_esval[:100]
        self.x_test = self.x_test[:100]
        self.y_test = self.y_test[:100]     
        self.epoch_count = 2
        
    
  def append_contributivity(self, contributivity):
            
      self.contributivity_list.append(contributivity)
          
  def split_data(self):
        """Populates the nodes with their train and test data (not pre-processed)"""
    
        #%% Fetch parameters of scenario
        
        x_train = self.x_train
        y_train = self.y_train
        x_test = self.x_test
        y_test = self.y_test
        
        # Describe data
        print('\n### Data loaded: ', self.dataset_name)
        print('- ' + str(len(x_train)) + ' train data with ' + str(len(y_train)) + ' labels')
        print('- ' + str(len(x_test)) + ' test data ' + str(len(y_test)) + ' labels')
        
        # Describe number of independant nodes
        print('\n### Description of data scenario configured:')
        print('- Number of nodes defined:', self.nodes_count)
        
        
        #%% Configure the desired splitting scenario - Datasets sizes
        # Should the nodes receive an equivalent amount of samples each...
        # ... or receive different amounts?
        
        # Check the percentages of samples per node and control its coherence
        assert(len(self.amounts_per_node) == self.nodes_count)
        assert(np.sum(self.amounts_per_node) == 1)
        
        # Then we parameterize this via the splitting_indices to be passed to np.split
        # This is to transform the % from my_scenario into indices where to split the data
        splitting_indices = np.empty((self.nodes_count-1,))
        splitting_indices[0] = self.amounts_per_node[0]
        for i in range(self.nodes_count-2):
            splitting_indices[i+1] = splitting_indices[i] + self.amounts_per_node[i+1]
        splitting_indices_train = (splitting_indices * len(y_train)).astype(int)
        splitting_indices_test = (splitting_indices * len(y_test)).astype(int)
        # print('- Splitting indices defined (for train data):', splitting_indices_train) # VERBOSE
        
        
        #%% Configure the desired data distribution scenario
        
        # Describe the type of distribution chosen
        print('- Data distribution scenario chosen:', self.samples_split_option)
        
        # Create a list of indexes of the samples
        train_idx = np.arange(len(y_train))
        test_idx = np.arange(len(y_test))
        
        # In the 'Stratified' scenario we sort MNIST by labels
        if self.samples_split_option == 'Stratified':
            
            # Sort MNIST by labels
            y_sorted_idx = y_train.argsort()
            y_train = y_train[y_sorted_idx]
            x_train = x_train[y_sorted_idx]
        
        # In the 'Random' scenario we shuffle randomly the indexes
        elif self.samples_split_option == 'Random':
            np.random.seed(42)
            np.random.shuffle(train_idx)
        
        # If neither 'Stratified' nor 'Random', we raise an exception
        else:
            raise NameError('This samples_split_option scenario [' + self.samples_split_option + '] is not recognized.')
            
            
        #%% Do the splitting among nodes according to desired scenarios
        
        # Split data between nodes
        train_idx_idx_list = np.split(train_idx, splitting_indices_train)
        test_idx_idx_list = np.split(test_idx, splitting_indices_test)
        
        # Describe test data distribution scenario
        print('- Test data distribution scenario chosen:', self.testset_option)
      
        # Populate nodes
        node_id = 0
        for train_idx, test_idx in zip(train_idx_idx_list, test_idx_idx_list):
            
            # Train data
            x_node_train = x_train[train_idx, :]
            y_node_train = y_train[train_idx,]
            
            # Test data
            if self.testset_option == 'Distributed':
                x_node_test = x_test[test_idx]
                y_node_test = y_test[test_idx]
            elif self.testset_option == 'Centralised':
                x_node_test = x_test
                y_node_test = y_test
            else:
                raise NameError('This testset_option [' + self.testset_option + '] scenario is not recognized')
                
            node = Node(x_node_train, x_node_test, y_node_train, y_node_test, str(node_id))
            self.node_list.append(node)
            node_id += 1
     
        # Check coherence of node_list versus nodes_count   
        assert(len(self.node_list) == self.nodes_count)
        
        # Print and plot for controlling
        print('\n### Splitting data among nodes:')
        for node_index, node in enumerate(self.node_list):
            print('- Node #' + str(node_index) + ':')
            print('  - Number of samples:' + str(len(node.x_train)) + ' train, ' + str(len(node.x_val)) + ' val, ' + str(len(node.x_test)) + ' test')
            print('  - y_train first 10 values:' + str(node.y_train[:10]))
            print('  - y_train last 10 values:' + str(node.y_train[-10:]))
            
        return 0
    
    
  def plot_data_distribution(self):
      
        for i, node in enumerate(self.node_list):
            
            plt.subplot(self.nodes_count, 1, i+1) #TODO share y axis
            # print(node.y_train) # VERBOSE
            #data = np.argmax(node.y_train, axis=1)
            data_count = np.bincount(node.y_train)
            
            #Fill with 0
            while len(data_count) < 10:
                data_count = np.append(data_count, 0)
                
            plt.bar(np.arange(0, 10), data_count)
            plt.ylabel('Node ' + str(i))
        
        plt.suptitle('Data distribution')          
        plt.xlabel('Digits')
        plt.savefig(self.save_folder / 'data_distribution.png')
      
      
  def to_file(self):
    
    out = ''
    out += 'Dataset name: ' + self.dataset_name + '\n'
    out += 'Number of data samples - train: ' + str(len(self.x_train)) + '\n'
    out += 'Number of data samples - test: ' + str(len(self.x_test)) + '\n'
    out += 'Nodes count: ' + str(self.nodes_count) + '\n'
    out += 'Percentages of data samples per node: ' + str(self.amounts_per_node) + '\n'
    out += 'Random or stratified split of data samples: ' + self.samples_split_option + '\n'
    out += 'Centralised or distributed test set: ' + self.testset_option + '\n'
    out += 'Number of epochs: ' + str(self.epoch_count) + '\n'
    out += 'Early stopping on? ' + str(self.is_early_stopping) + '\n'
    out += 'Test score of federated training: ' + str(self.federated_test_score) + '\n'
    out += '\n'
    
    out += str(len(self.contributivity_list)) + ' contributivity methods: ' + '\n'

    for contrib in self.contributivity_list:
      out += str(contrib) + '\n\n'
       

    target_file_path = self.save_folder / 'results_summary.txt'
    
    with open(target_file_path, 'w', encoding='utf-8') as f:
        f.write(out)
