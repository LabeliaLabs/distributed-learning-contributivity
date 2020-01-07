# -*- coding: utf-8 -*-
"""
This enables to parameterize the contributivity measurements to be performed.
"""

import datetime


class Contributivity:
  def __init__(self, name='', contributivity_scores=[], computation_time=0):
    self.name = name
    self.contributivity_scores = contributivity_scores
    self.computation_time = computation_time
    
    
  def __str__(self):
    output = self.name + '\n'
    output += 'computation time: ' + str(datetime.timedelta(seconds=self.computation_time)) + '\n'
    #TODO print only 3 digits
    output += 'contributivity scores: ' + str(self.contributivity_scores) 
      
    return output

