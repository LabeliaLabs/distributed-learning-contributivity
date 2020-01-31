# -*- coding: utf-8 -*-
"""
This enables to parameterize the contributivity measurements to be performed.
"""

import datetime
import numpy as np


class Contributivity:
    def __init__(self, name='', contributivity_scores=np.array([]), scores_variances=np.array([]), computation_time=0):
        self.name = name
        n=len(contributivity_scores)
        assert(n==len(contributivity_scores[i] ) )
        self.contributivity_scores = contributivity_scores
        self.scores_variances = scores_variances

        self.computation_time = computation_time


    def __str__(self):
        output = '\n' + self.name + '\n'
        output += 'computation time: ' + str(datetime.timedelta(seconds=self.computation_time)) + '\n'
        #TODO print only 3 digits
        output += 'contributivity scores: ' + str(self.contributivity_scores)+ '\n'
        output += 'variances of the contributivity scores: ' + str(self.scores_variances)+ '\n'


        return output

