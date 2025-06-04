"""
This module contains the Dysglycemia() class (component) used to handle transitions from Ab1 and mAb1 to dysglycemic state

Methods:
    setup method: where the component initialization takes place. The simulation engine looks for a setup method on each component and calls that method passing a Builder instance as a parameter 

    registers ab1_to_dysglycemia_rate and mab1_to_dysglycemia_rate onto engine Value System. Later, in other components, we might register value modifiers that act upon this rate

    base_ab1_to_dysglycemia_rate: this method computes transition probabilites for simulants passed in pd.index

    base_mab1_to_dysglycemia_rate: this method computes transition probabilites for simulants passed in pd.index

    determine_mab1_to_dysglycemia: the two methods above computes individual transition probabilities. This method uses those to handle transition logic. Who transitions who remains



Note:
    1) This module requires two key simulation engine systems: Builder, Event, which must be imported from vivarium.framework
    2) 2) This modules requires external serialized survival regression models (compute transition probabilities) to be imported, loaded with pickle and used in 
        base_ab1_to_dysglycemia_rate and base_mab1_to_dysglycemia_rate and 
"""

import pandas as pd
import pickle
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event

######## GET BINARY FILE ###########
# ab1 to dysglycemic
with open("transition_probabilities/binary_files/sAB2Hyperglycemia.bin","rb") as binary_ab1_to_dysglycemic:
    ab1_to_dysglycemic_model = pickle.load(binary_ab1_to_dysglycemic)
      
# mab1 to dysglycemic
with open("transition_probabilities/binary_files/mAB2Hyperglycemia.bin","rb") as binary_mab1_to_dysglycemic:
    mab1_to_dysglycemic_model = pickle.load(binary_mab1_to_dysglycemic)
   

class Dysglycemia:
    def __init__(self):
        self.name = "dysglycemia"
        self.completed_cycles = 0

    def setup(self, builder: Builder):
        """this is wehre the component initialization takes place"""
        self.population_view = builder.population.get_view(["state","previous_state","time_in_state","GRS2","fdr","age"])

        self.ab1_population_view = builder.population.get_view(["state", "previous_state", "time_in_state", "GRS2", "fdr", "age"], query="state == 'Ab1'")
        self.mab1_population_view = builder.population.get_view(["state", "previous_state", "time_in_state", "GRS2", "fdr", "age"], query="state == 'mAb1'")

        self.ab1_to_dysglycemia_rate = builder.value.register_rate_producer("ab1_to_dysglycemia_rate", source=self.base_ab1_to_dysglycemia_rate)
        self.mab1_to_dysglycemia_rate = builder.value.register_rate_producer("mab1_to_dysglycemia_rate", source=self.base_mab1_to_dysglycemia_rate)

        self.randomness_ab1_to_dysglycemia = builder.randomness.get_stream("ab1_dysglycemia")
        self.randomness_mab1_to_dysglycemia = builder.randomness.get_stream("mab1_dysglycemia")

        builder.event.register_listener("time_step", self.determine_ab1_to_dysglycemia)
        builder.event.register_listener("time_step", self.determine_mab1_to_dysglycemia)
        builder.event.register_listener("time_step", self.determine_time_in_state)
        

    def base_ab1_to_dysglycemia_rate(self, index: pd.Index) -> pd.Series:
        """get individual transition probabilities"""
        population_df = self.ab1_population_view.get(index)
        survival_prob = ab1_to_dysglycemic_model.predict_survival_function(population_df, times=1, conditional_after=population_df['time_in_state'])
        rate = 1 - survival_prob.iloc[0]
        return rate

    def base_mab1_to_dysglycemia_rate(self, index: pd.Index) -> pd.Series:
        """get individual transition probabilities"""
        population_df = self.mab1_population_view.get(index)
        survival_prob = mab1_to_dysglycemic_model.predict_survival_function(population_df, times=1, conditional_after=population_df['time_in_state'])
        rate = 1 - survival_prob.iloc[0]
        return rate
    
    def _get_ab1_population_index(self, ab1_population):
        ab1_index = ab1_population.index
        return ab1_index
    
    def _compute_affected_individuals_ab1_to_dysglycemia(self, effective_ab1_to_dysglycemia, ab1_index):
        """compute affected individuals"""
        draw_ab1_to_dysglycemia = self.randomness_ab1_to_dysglycemia.get_draw(ab1_index)
        affected_ab1_to_dysglycemia = (draw_ab1_to_dysglycemia < effective_ab1_to_dysglycemia)
        return affected_ab1_to_dysglycemia
    
    def _get_mab1_population_index(self, mab1_population):
        mab1_index = mab1_population.index
        return mab1_index
    
    def _compute_affected_individuals_mab1_to_dysglycemia(self, effective_mab1_to_dysglycemia, mab1_index):
        """compute affected individuals"""
        draw_mab1_to_dysglycemia = self.randomness_mab1_to_dysglycemia.get_draw(mab1_index)
        affected_mab1_to_dysglycemia = (draw_mab1_to_dysglycemia < effective_mab1_to_dysglycemia)
        return affected_mab1_to_dysglycemia
    

    
    def determine_ab1_to_dysglycemia(self, event: Event):
        """Determines who transitions or not"""
        self.completed_cycles += 1

        if self.completed_cycles < 2:
            return
        
        ab1_population = self.ab1_population_view.get(event.index)
        ab1_index = self._get_ab1_population_index(ab1_population)
        effective_ab1_to_dysglycemia = self.ab1_to_dysglycemia_rate(ab1_index)
        affected_ab1_to_dysglycemia = self._compute_affected_individuals_ab1_to_dysglycemia(effective_ab1_to_dysglycemia, ab1_index)
        self.population_view.update(pd.Series("dysglycemic", index=ab1_index[affected_ab1_to_dysglycemia], name ="state"))

    def determine_mab1_to_dysglycemia(self, event: Event):
        """Determines who transitions or not"""
        if self.completed_cycles < 2:
            return
        
        mab1_population = self.mab1_population_view.get(event.index)
        mab1_index = self._get_mab1_population_index(mab1_population)
        effective_mab1_to_dysglycemia = self.mab1_to_dysglycemia_rate(mab1_index)
        affected_mab1_to_dysglycemia = self._compute_affected_individuals_mab1_to_dysglycemia(effective_mab1_to_dysglycemia, mab1_index)
        self.population_view.update(pd.Series("dysglycemic", index=mab1_index[affected_mab1_to_dysglycemia], name= "state"))


    def determine_time_in_state(self, event: Event):
        """
        Updates the 'time_in_state' and 'previous_state' for individuals in the population.
        Args:
            event (Event): The event containing the index of the population to be evaluated.
        Returns:
            None
        """
        # get population
        population = self.population_view.get(event.index)
        # get all rows in the population where state != previous state
        # these individuals just changed state in this cycle, so reset their time_in_state to 0
        population.loc[population["state"] != population["previous_state"], "time_in_state"] = 0
        # update previous_state to the current state for those who changed state
        population.loc[population["state"] != population["previous_state"], "previous_state"] = population.loc[population["state"] != population["previous_state"], "state"]
        # update the population view with the modified data
        self.population_view.update(population)
