"""
Chat with Lauric -> 2 interventions:

     1) screening (only effect is to change risk of DKA) -----> focus on this one for now

     2) you take a drug (teplizumab) delays T1D onset
     
     Critical to learn:

     1) randomness system 
        (we want randomness to be consistent across baseline and intervention simulations)
     2) value system 
        in vivarium, values are rates or probabilities that are influenced by multiple components and are dynamic (change over time)
     
     for now, intervention will have an effect on reducing with DKA envents
     """


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.interface import InteractiveContext



class ScreeningIntervention():
     
    configuration_defaults = {
        "screening_intervention": {
            "effect_size": 0.1,  # reduction in our affected value (t1d rate)
        }
    }

    def __init__(self, name: str, affected_value: str):
        self.name = name
        self.affected_value = affected_value
        self.configuration_defaults = ScreeningIntervention.configuration_defaults
        self.completed_cycles = 0
        self.affected_simulants = []

      
  

    def setup(self, builder:Builder):
        self.population_view = builder.population.get_view(["screen_status"])
   

        #register funtion that modifies an existing value (T1D with DKA in our case)
        builder.value.register_value_modifier("further_t1d_splitting_rate", modifier=self.intervention_effect,
                                              requires_columns=["screen_status"])

        #resgister cycle count as listener for events
        builder.event.register_listener("time_step", self.updated_cycle_count)
        
    def updated_cycle_count(self, event: Event):
        self.completed_cycles = self.completed_cycles + 1

    def intervention_effect(self, index, value):

        if self.completed_cycles < 2:
            return value
        else:
            screen_status = self.population_view.get(index)['screen_status']
            condition = screen_status.isin(['Ab1', 'mAb1', 'dysglycemic','type1_diabetes'])
            affected_indices = index[condition]

            value.loc[affected_indices] = value.loc[affected_indices] * 0.119

            return value
            

    

                








# class ScreeningIntervention():
     
#     configuration_defaults = {
#         "screening_intervention": {
#             "effect_size": 0.1,  # reduction in our affected value (t1d rate)
#         }
#     }

#     def __init__(self, name: str, affected_value: str):
#         self.name = name
#         self.affected_value = affected_value
#         self.configuration_defaults = ScreeningIntervention.configuration_defaults
#         self.completed_cycles = 0
#         self.affected_simulants = []

      
  

#     def setup(self, builder:Builder):
#         self.population_view = builder.population.get_view(["screen_status"])
   

#         #register funtion that modifies an existing value (T1D with DKA in our case)
#         builder.value.register_value_modifier("further_t1d_splitting_rate", modifier=self.intervention_effect,
#                                               requires_columns=["screen_status"])

#         #resgister cycle count as listener for events
#         builder.event.register_listener("time_step", self.updated_cycle_count)
        
#     def updated_cycle_count(self, event: Event):
#         self.completed_cycles = self.completed_cycles + 1

#     def intervention_effect(self, index, value):

#         if self.completed_cycles < 2:
#             return value
#         else:
#             screen_status = self.population_view.get(index)['screen_status']
#             condition = screen_status.isin(['Ab1', 'mAb1', 'dysglycemic','type1_diabetes'])
#             affected_indices = index[condition]

        
#             value.loc[affected_indices] = value.loc[affected_indices] * 0.119
#             #value.loc[affected_indices] = value.loc[affected_indices] * 0.01



#             return value
            

    

                
