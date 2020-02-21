# -*- coding: utf-8 -*-
"""
This enables to parameterize the contributivity measurements to be performed.
"""

import datetime
import numpy as np


class Contributivity:
    def __init__(self, name='', contributivity_scores=np.array([]), scores_std=np.array([]), computation_time=0):
        self.name = name
        self.contributivity_scores = contributivity_scores
        self.scores_std = scores_std
        sum_contrib=np.sum(contributivity_scores)
        if len (contributivity_scores)==0:
            elf.normalized_scores = np.array([])
        else:
            self.normalized_scores = contributivity_scores/sum_contrib
        self.computation_time = computation_time


    def __str__(self):
        output = '\n' + self.name + '\n'
        output += 'Computation time: ' + str(datetime.timedelta(seconds=self.computation_time)) + '\n'
        #TODO print only 3 digits
        output += 'Contributivity scores: ' + str(self.contributivity_scores)+ '\n'
        output += 'Std of the contributivity scores: ' + str(self.scores_std)+ '\n'
        output += 'Normalized contributivity scores: ' + str(self.normalized_scores)+ '\n'


        return output

