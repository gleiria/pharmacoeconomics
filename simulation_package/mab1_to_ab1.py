"""
This module contains the MultiToAutoInsideAutoantibody class (component) which handles transitions from mAb1 to Ab1 state

Methods:
    setup method: automatically called by the simulation egine which passes in a ``builder`` object which provides access to a variety of framework subsystems and metadata.

                registers mab1_to_ab1_rate in engine Value System. Later, in other components, we might register value modifiers that act upon this rate

    base_mab1_to_ab1_transition_rate: this method computes transition probabilites for simulants passed in pd.index


    determine_mab1_to_ab1: handles transition logic and uses transitions computed by the two methods above


Example usage:
    >>> from simulation_package.ab1_to_mab1 import MultiToAutoInsideAutoantibody
    >>> pass it as an argument to InteractiveContext(components)

Note:
    1) This module requires two key simulation engine systems: Builder and which must be imported from vivarium.framework
    2) This modules requires external serialized survival regression models (compute transition probabilities) to be imported, loaded with pickle and used in 
        base_mab1_to_ab1_transition_rate. 
"""

import pandas as pd
import pickle
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from vivarium.framework.engine import Builder
from vivarium.framework.event import Event

######## GET BINARY FILE ###########
# mab1 to ab1
with open("transition_probabilities/binary_files/mAB2sAB.bin","rb") as binary_mab1_to_ab1:
    mab1_to_ab1_model = pickle.load(binary_mab1_to_ab1)

class MultiToAutoInsideAutoantibody:

    def __init__(self):
        self.name = "mab1_to_ab1_intermediate"
        self.completed_cycles = 0
        self.survival_probs_mab1_to_ab1 = []

    def setup(self, builder: Builder):
        self.population_view = builder.population.get_view(["state","previous_state","time_in_state","GRS2","fdr","age"])
        self.mab1_population_view = builder.population.get_view(["state"], query="state == 'mAb1'")

        self.mab1_to_ab1_rate = builder.value.register_rate_producer("mab1_to_ab1_rate", source=self.base_mab1_to_ab1_transition_rate)
        self.mab1_to_ab1_randomness = builder.randomness.get_stream("autoantibody_3")
        builder.event.register_listener("time_step", self.determine_mab1_to_ab1)
        builder.event.register_listener("time_step", self.determine_time_in_state)

    def base_mab1_to_ab1_transition_rate(self, index: pd.Index) -> pd.Series:
        population_df = self.population_view.get(index)
        survival_prob = mab1_to_ab1_model.predict_survival_function(population_df, times=1, conditional_after=population_df['time_in_state'])
        self.survival_probs_mab1_to_ab1.append(survival_prob)
        rate = 1 - survival_prob.iloc[0]
        return rate
    
    def _get_mab1_population_index(self, full_population):
        mab1_population = full_population[full_population["state"]== "mAb1"]
        mab1_population_index = mab1_population.index
        return mab1_population_index
    
    def _compute_affected_individuals(self, effective_mab1_to_ab1_rate, mab1_index):
        draw_mab1_to_ab1 = self.mab1_to_ab1_randomness.get_draw(mab1_index)
        affected_mab1_to_ab1 = draw_mab1_to_ab1 < effective_mab1_to_ab1_rate
        return affected_mab1_to_ab1


    def determine_mab1_to_ab1(self, event: Event):
        self.completed_cycles += 1
        if self.completed_cycles < 2:
            return
        
        full_population = self.population_view.get(event.index)
        mab1_index = self._get_mab1_population_index(full_population)
        effective_mab1_to_ab1_rate = self.mab1_to_ab1_rate(mab1_index)
        affected_mab1_to_ab1 = self._compute_affected_individuals(effective_mab1_to_ab1_rate, mab1_index)
        self.population_view.update(pd.Series("Ab1", index=mab1_index[affected_mab1_to_ab1], name="state"))

    def determine_time_in_state(self, event:Event):
        population = self.population_view.get(event.index)
        population.loc[population["state"] != population["previous_state"], "time_in_state"] = 0
        population.loc[population["state"] != population["previous_state"], "previous_state"] = population.loc[population["state"] != population["previous_state"], "state"]
        self.population_view.update(population)
        








