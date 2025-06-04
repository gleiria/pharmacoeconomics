"""
This module contains the AutoToMultiInsideAutoAntibody class (component) which handles transitions from Ab1 to mAb1 state

Methods:
    setup method: automatically called by the simulation egine which passes in a ``builder`` object which provides access to a variety of framework subsystems and metadata.

                registers  ab1_to_mab1_rate into the Value System. Later, in other components, we might register value modifiers that act upon these rates

    base_ab1_to_mab1_rate: this method computes transition probabilites for simulants passed in pd.index

    determine_ab1_to_mab1: handles transition logic and uses transitions computed by the two methods above


Example usage:
    >>> from simulation_package.ab1_to_mab1 import AutoToMultiInsideAutoAntibody
    >>> pass it as an argument to InteractiveContext(components)

Note:
    1) This module requires two key simulation engine systems: Builder and which must be imported from vivarium.framework
    2) This modules requires external serialized survival regression models (compute transition probabilities) to be imported, loaded with pickle and used in 
        base_ab1_to_mab1_transition_rate. 
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
import pandas as pd
import pickle

######## GET BINARY FILE ###########
# ab1 to mab1
with open("transition_probabilities/binary_files/sAB2mAB.bin","rb") as binary_ab1_to_mab1:
    ab1_to_mab1_model = pickle.load(binary_ab1_to_mab1)


class AutoToMultiInsideAutoAntibody:
    """ Class handles transitions from Ab1 to mAb1 and updates time in state"""

    def __init__(self):
        self.name = "ab1_to_mab1_intermediate"
        self.completed_cycles = 0

    def setup(self, builder: Builder):
        """this is called by Vivarium and makes registrations in both directions. Component to Vivarium and vive-versa"""

        self.population_view = builder.population.get_view(["state","previous_state","time_in_state","GRS2","fdr","age"])
        self.ab1_population_view = builder.population.get_view(["state"], query="state == 'Ab1'")

        # register value producer
        self.ab1_to_mab1_rate = builder.value.register_rate_producer("ab1_to_mab1_rate", source=self.base_ab1_to_mab1_transition_rate)

        # register in randomness system
        self.ab1_to_mab1_randomness = builder.randomness.get_stream("ab1_to_mab1_randomness")

        # Register event-driven methods
        builder.event.register_listener("time_step", self.determine_ab1_to_mab1)
        builder.event.register_listener("time_step", self.determine_time_in_state)

    def base_ab1_to_mab1_transition_rate(self, index: pd.Index) -> pd.Series:
        """this method computes transition pob and is called by determine_ab1_to_mab1() method"""
        population_df = self.population_view.get(index)
        survival_prob = ab1_to_mab1_model.predict_survival_function(population_df, times=1, conditional_after=population_df['time_in_state'])
        rate = 1 - survival_prob.iloc[0]
        return rate
    
    def _get_ab1_population_index(self, ab1_population):
        ab1_population_index = ab1_population.index
        return ab1_population_index
    
    def _compute_affected_individuals(self, effective_ab1_to_mab1_rate, ab1_population_index):
        # random draw
        draw_ab1_to_mab1 = self.ab1_to_mab1_randomness.get_draw(ab1_population_index)
        affected_ab1_to_mab1 = draw_ab1_to_mab1 < effective_ab1_to_mab1_rate
        return affected_ab1_to_mab1


    def determine_ab1_to_mab1(self, event: Event):
        """Event-Driven Method"""

        self.completed_cycles += 1

        if self.completed_cycles < 2:
            return
        # grabbing Ab1 population
        ab1_population = self.ab1_population_view.get(event.index)
        ab1_population_index = self._get_ab1_population_index(ab1_population)

        # using base_ab1_to_mab1_transition_rate() method to compute transitions 
        effective_ab1_to_mab1_rate = self.ab1_to_mab1_rate(ab1_population_index)

        # use helper function _compute_affected_individuals to grab who transitions
        affected_ab1_to_mab1 = self._compute_affected_individuals(effective_ab1_to_mab1_rate, ab1_population_index)

        # update
        self.population_view.update(pd.Series("mAb1", index=ab1_population_index[affected_ab1_to_mab1], name="state"))

    def determine_time_in_state(self, event:Event):
        """Event-Driven Method"""
        # retrieve current state of all simulants
        population = self.population_view.get(event.index)
        # get all rows in the population where state != previous state ---> these guys just changed state in this cycle ---> change their time_in_state to 1
        population.loc[population["state"] != population["previous_state"], "time_in_state"] = 0
        # grab same rows and make previous state to be state
        population.loc[population["state"] != population["previous_state"], "previous_state"] = population.loc[population["state"] != population["previous_state"], "state"]
        self.population_view.update(population)


