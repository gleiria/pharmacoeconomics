

import os
import sys

from scipy.stats import norm
import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class Screening:
    def __init__(self, continuous_vector=None):
        self.name = "screening"
        self.completed_cycles = 0

        # distribution parameters
        self.mean = 10
        self.std_dev = 2.375

        self.threshold_vector = self.compute_threshold_vector(continuous_vector)

    def compute_threshold_vector(self, continuous_vector):
        grs_threshold_vector = []
        for percentile in continuous_vector:
            if percentile == 0:
                threshold = float("inf")
            elif percentile == 1:
                threshold = float("-inf")
            else:
                threshold = norm.ppf(percentile, loc=self.mean, scale=self.std_dev)
            grs_threshold_vector.append(threshold)
        return grs_threshold_vector

    def setup(self, builder: Builder):
        self.population_view = builder.population.get_view(["age","number_of_screens","GRS2","screen_status","screened_in_past","state", "screening_cost",])
        self.screening_rate = builder.value.register_rate_producer("screening_rate", source=self.base_screening_rate)
        self.randomness = builder.randomness.get_stream("screening_randomness")
        builder.event.register_listener("time_step", self.determine_screening)

    def base_screening_rate(self, index: pd.Index) -> pd.Series:
        return pd.Series(1, index=index)

    def determine_screening(self, event: Event):
        self.completed_cycles += 1
        threshold = self.threshold_vector[self.completed_cycles - 1]

        population = self.population_view.get(event.index)

        if threshold == float("inf"):
            return
        elif threshold == float("-inf"):
            eligible_for_screening = population
        else:
            eligible_for_screening = population.loc[population["GRS2"] > threshold]

        # if the eligible_for_screening dataframe is not empty: 
        if not eligible_for_screening.empty:
            population_index = eligible_for_screening.index
            effective_screening_rate = self.screening_rate(population_index)
            draw = self.randomness.get_draw(population_index)
            affected_screening = draw <= effective_screening_rate

            # Using affected indices directly for loc
            affected_indices = population_index[affected_screening]
          
            
            # if there are individuals for screening
            if affected_screening.any():
                # 1) update screened_in_past collumn
                self.population_view.update(pd.Series(1, index=affected_indices, name="screened_in_past"))
                
                # 2) update screen_status column
                update_conditions = eligible_for_screening['state'].isin(['Ab1', 'mAb1', 'dysglycemic','type1_diabetes'])
                filtered_affected_indices = affected_indices[update_conditions[affected_indices]]
                screen_status_update = pd.Series(eligible_for_screening.loc[filtered_affected_indices, "state"],index=filtered_affected_indices,name="screen_status")
                if not screen_status_update.empty:
                    self.population_view.update(screen_status_update)
                # 3) update number_of_screens collumn
                number_of_screens_update = pd.Series(eligible_for_screening.loc[affected_indices, "number_of_screens"]+ 1,index=affected_indices,name="number_of_screens",)
                self.population_view.update(number_of_screens_update)

                current_costs = eligible_for_screening.loc[affected_indices, "screening_cost"]
              
                updated_costs = current_costs + 92
                self.population_view.update(
                    pd.Series(
                        updated_costs, index=affected_indices, name="screening_cost"
                    )
                )

