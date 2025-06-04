import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class Type1DiabetesDkaSplitting:

    def __init__(self, dka_ratio=None):
        self.name = "type_1_diabetes_dka_splitting"
        self.completed_cycles = 0
        self.dka_ratio = dka_ratio

    def setup(self, builder:Builder):
        self.population_view = builder.population.get_view(["state","previous_state","time_in_state","GRS2","fdr","age","screen_status","t1d_cost"])
        self.t1d_population_view = builder.population.get_view(["state","previous_state","time_in_state","GRS2","fdr","age","t1d_cost"], query = "state == 'type1_diabetes'")
        

        self.further_t1d_splitting_rate = builder.value.register_rate_producer("further_t1d_splitting_rate", source=self.base_further_t1d_splitting_rate)
        self.further_t1d_splitting_randomness = builder.randomness.get_stream("further_t1d_splitting")

        # register determine_t1d in event system
        builder.event.register_listener("time_step", self.determine_t1d)

    def base_further_t1d_splitting_rate(self, index:pd.Index) -> pd.Series:
        rate = pd.Series(self.dka_ratio, index=index)
        return rate
    
    def _compute_future_state(self, effective_further_t1d_rate, t1d_population_index):
        # calculate transitions
        draw_further_t1d_rate = self.further_t1d_splitting_randomness.get_draw(t1d_population_index)
        affected_indices = (draw_further_t1d_rate < effective_further_t1d_rate)

        further_splitting = pd.Series(np.where(affected_indices, "T1D_with_DKA", "T1D_without_DKA"), index=t1d_population_index, name="state")
        
        return further_splitting
    
    def determine_t1d(self, event:Event):
        self.completed_cycles += 1

        if self.completed_cycles < 2:
            return
        
        # get T1D population
        t1d_population = self.t1d_population_view.get(event.index)
        # index from above
        t1d_population_index = t1d_population.index

        # calculate transitions
        effective_further_t1d_rate = self.further_t1d_splitting_rate(t1d_population_index)

        further_splitting = self._compute_future_state(effective_further_t1d_rate, t1d_population_index)
        self.population_view.update(further_splitting)
        
        # t1d monitoring costs. 
        # Here I didn't create a dedicated method because I reallyu need to make sure only further_splitting individuals are updated.
        t1d_costs = pd.Series(np.where(further_splitting == "T1D_with_DKA", 15077, 15077), index=t1d_population_index, name="t1d_cost")
        self.population_view.update(t1d_costs)

            

           