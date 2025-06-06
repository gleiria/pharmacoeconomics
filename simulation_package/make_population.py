"""
This module contains the Population() class (component) used to create a population of simulants.


Methods:
    setup method: where most of the component initialization takes place. The simulation engine looks for a setup method on each component and calls that method with a Builder instance (input)

    on_initializes_simulants: this method is registered in the setup() method as a simulant initializer. With current implementation this happens a single time at the very beginning of a simulation.
                              like the setup method, it takes an argument passed automatically by the simulation engine. pop_data is an instance of SimulantData which has information useful when initializing simulants.

    age_simulants: this method, surprise surprise, ages simulants at every simulation cycle. The method is a listener of the "time_step" event so that it fires in every cycle. 
                   this method, taking advantage that it is a listener of the cycle event, also handles the counting of number of years in a particular state which is later used by the external survival regression models.


Example usage:
    >>> from simulation_package.make_population import Population
    >>> pass it as an argument to InteractiveContext(components)

Note:
    This module requires three key simulation engine systems: Builder, Event, SimulantData which must be imported from vivarium.framework
"""
import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

import pandas as pd
import numpy as np


#wtcc dataset to compute GRS
df_wtcc = pd.read_csv("background_population_grs.csv")
df_wtcc = df_wtcc.drop(columns= "Unnamed: 0")
no_t1d_wtcc_df = df_wtcc[df_wtcc["t1d_status"]== 0]
no_t1d_wtcc_list = no_t1d_wtcc_df["GRS2"].tolist()

class Population:
    """
    Component creates population of simulants
    """

    configuration_defaults = {
        "population": {
            "age_start": 0,
            "time_in_state":0,
        },
    }

    def __init__(self):
        self.name = "population"
        self.completed_cycles = 0
        self.no_t1d_wtcc_list = no_t1d_wtcc_list
    

    def setup(self, builder: Builder):
        """
        Set up Population component
        """
        self.config = builder.configuration
        columns_created = ["age","GRS2", "fdr", "state","time_in_state","previous_state","screened_in_past","number_of_screens","screen_status","screening_cost", "market_basket_cost",
                           "t1d_cost","entrance_time", "ever_antibody"]
        # testing grs into randomness system
        self.grs_randomness = builder.randomness.get_stream("grs_initialization")
        self.family_history_randomness = builder.randomness.get_stream("family_history_initialization")
        self.screening = builder.randomness.get_stream("screening_initialization")
        self.screening_cost_randmoness = builder.randomness.get_stream("screening_cost_initialization")
        self.t1d_cost_randmoness = builder.randomness.get_stream("t1d_cost_initialization")
        self.ever_antibody_randomness = builder.randomness.get_stream("ever_antibody_initialization")
        
        
        builder.population.initializes_simulants(self.on_initializes_simulants, creates_columns=columns_created)

        self.population_view = builder.population.get_view(columns_created)
        builder.event.register_listener("time_step", self.age_simulants)

    def on_initializes_simulants(self, pop_data: SimulantData):

        """
        Initialize simulants in the population
        """
        #determine family history probabilities and grs 
        family_history_probs = self.family_history_randomness.get_draw(pop_data.index)
        #uniform_randoms = self.grs_randomness.get_draw(pop_data.index)
        grs_random_draws = self.grs_randomness.get_draw(pop_data.index)
        mapped_indices = (grs_random_draws * len(self.no_t1d_wtcc_list)).astype(int)
        grs2_values = [self.no_t1d_wtcc_list[idx] for idx in mapped_indices]



        population = pd.DataFrame(
            {
                "age": self.config.population.age_start,
                "GRS2":grs2_values, # change this to 0 - 20 normally distributed (mean of 10 and sigma value 2.375)
                "fdr":np.where(family_history_probs < 0.02, 1,0), # 1 = yes, 0 = no
                "state": pd.Series("healthy", index=pop_data.index),
                "previous_state": pd.Series("healthy", index = pop_data.index),
                "time_in_state": self.config.population.time_in_state,
                "screened_in_past": pd.Series(0, index= pop_data.index), # 1 = yes, 0 = no
                "number_of_screens": pd.Series(0, index= pop_data.index),
                "screen_status": pd.Series("not_screened", index=pop_data.index),
                "screening_cost": pd.Series(0, index=pop_data.index),
                "t1d_cost": pd.Series(0, index = pop_data.index),
                "market_basket_cost": pd.Series(0, index = pop_data.index),
                "entrance_time":pop_data.creation_time,
                "ever_antibody":pd.Series(0, index= pop_data.index) # 1 = yes, 0 = no,
            },
            index=pop_data.index,
        )
        self.population_view.update(population)

    def age_simulants(self, event: Event):
        """
        age simulants in the population + time in state
        """

        self.completed_cycles +=1
        population = self.population_view.get(event.index)
      

        population.loc[population["state"] == population["previous_state"], "time_in_state"] += 1
        
        
        population["age"] += 1
        self.population_view.update(population)

   
        
