"""
This module contains the AutoAntibody() class (component) which handles transitions from healthy to Ab1 and mAb1 states 


Methods:
    setup method: automatically called by the simulation egine which passes in a ``builder`` object which provides access to a variety of framework subsystems and metadata.
                registers two values in the engine "Value" system: ab1_rate and mab1_rate. Later, in other components, we might register value modifiers that act upon these rates

    base_ab1_transition_rate: this method computes transition probabilites for simulants passed in pd.index

    base_mab1_transition_rate: this method computes transition probabilites for simulants passed in pd.index

    determine_autoantibody: handles transition logic and uses transitions computed by the two methods above


Example usage:
    >>> from simulation_package.make_population import AutoAntibody()
    >>> pass it as an argument to InteractiveContext(components)

Note:
    1) This module requires three key simulation engine systems: Builder, Event, SimulantData which must be imported from vivarium.framework
    2) This modules requires external serialized survival regression models (compute transition probabilities) to be imported, loaded with pickle and used in 
        base_ab1_transition_rate and mbase_ab1_transition_rate. 
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import pickle

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

######## GET BINARY FILES ###########

# healthy to ab1
with open(
    "transition_probabilities/binary_files/healthy2sAB.bin", "rb") as binary_healthy_to_ab1:
    healthy_to_ab1_model = pickle.load(binary_healthy_to_ab1)


# healthy to mab1
with open(
    "transition_probabilities/binary_files/healthy2mAB.bin", "rb") as binary_healthy_to_mab1:
    healthy_to_mab1_model = pickle.load(binary_healthy_to_mab1)


class AutoAntibody:
    """
    Component handles transitions from healthy to Ab1 and mAb1 states.
    """
    
    def __init__(self):
        self.name = "first_intermediate"
        self.completed_cycles = 0
        self.survival_probs_ab1 = []
        self.survival_probs_mab1 = []

    def setup(self, builder: Builder):
           """
        The setup method gives the component access to an instance of the Builder which exposes a handful of tools to help build components.
        The simulation framework is responsible for calling the setup method on components and providing the builder to them

        Args:
            builder (Builder): 
        """
        self.population_view = builder.population.get_view(
            [
                "state",
                "previous_state",
                "time_in_state",
                "GRS2",
                "fdr",
                "market_basket_cost",
                "ever_antibody",
            ]
        )
        # get ages
        self.age_view = builder.population.get_view(["age"])
        # get grs(s)
        self.grs_view = builder.population.get_view(["GRS2"])
        # get family history
        self.family_history_view = builder.population.get_view(["fdr"])
        # get healthy simulants
        self.healthy_population_view = builder.population.get_view(
            ["state"], query="state == 'healthy'"
        )

        #### register value producers ####
        self.ab1_rate = builder.value.register_rate_producer(
            "ab1_rate", source=self.base_ab1_transition_rate
        )
        self.mab1_rate = builder.value.register_rate_producer(
            "mab1_rate", source=self.base_mab1_transition_rate
        )
        self.randomness = builder.randomness.get_stream("autoantibody")

        # register these 3 public methods within Vivarium's Event system so that they fire at each time step
        builder.event.register_listener("time_step", self.determine_autoantibody)
        builder.event.register_listener("time_step", self.determine_ever_antibody)
        builder.event.register_listener("time_step", self.determine_time_in_state)

    def base_ab1_transition_rate(self, index: pd.Index) -> pd.Series:
        population_df = self.population_view.get(index)
        survival_prob = healthy_to_ab1_model.predict_survival_function(population_df, times=1, conditional_after=population_df["time_in_state"])
        # Store the surv probs for each cycle
        self.survival_probs_ab1.append(survival_prob)
        rate = 1 - survival_prob.iloc[0]
        rate = rate.where(population_df["GRS2"] >= 8.2, rate / 2.6)  # if condition evaluates to True, divide
        rate = rate.where(population_df["GRS2"] >= 9.7, rate / 3.2)
        rate = rate.where(population_df["GRS2"] >= 11, rate / 3.4)
        rate = rate.where(population_df["GRS2"] >= 12.4, rate / 3.89)  # therefore GRS above 12.4 won't be touched
        return rate

    def base_mab1_transition_rate(self, index: pd.Index) -> pd.Series:
        population_df = self.population_view.get(index)
        survival_prob = healthy_to_mab1_model.predict_survival_function(population_df, times=1, conditional_after=population_df["time_in_state"])
        self.survival_probs_mab1.append(survival_prob)
        rate = 1 - survival_prob.iloc[0]
        rate = rate.where(population_df["GRS2"] >= 8.2, rate / 2.6)
        rate = rate.where(population_df["GRS2"] >= 9.7, rate / 3.2)
        rate = rate.where(population_df["GRS2"] >= 11, rate / 3.4)
        rate = rate.where(population_df["GRS2"] >= 12.4, rate / 3.89)
        return rate

    def _get_healthy_population(self, full_population):
        """
        Helper function to filter and return the indices of healthy simulants.
        Parameters:
        full_population (pd.DataFrame): The DataFrame containing the full population data.
        Returns:
        pd.Index: The index of healthy simulants.
        """
        # Filter the population to get only healthy simulants
        healthy_population = full_population[full_population["state"] == "healthy"]
        # Extract and return the index of healthy simulants
        healthy_index = healthy_population.index
        return healthy_index

    def _compute_future_state(self, effective_ab1_rate, effective_mab1_rate, healthy_index):
        """
        Helper function that computes future states and puts ab1 and mab1 into a pd.Series needed to update back to Vivarium.
        Parameters:
        effective_ab1_rate (pd.Series): The computed transition rates for simulants from healthy to Ab1 state.
        effective_mab1_rate (pd.Series): The computed transition rates for simulants from healthy to mAb1 state.
        healthy_index (pd.Index): The index of healthy simulants.

        Returns:
        ab1_series (pd.Series): Series indicating simulants transitioning to Ab1 state.
        mab1_series (pd.Series): Series indicating simulants transitioning to mAb1 state.
        """
        # Merge transition rates into a single DataFrame
        merged_df = pd.concat([effective_ab1_rate.rename("ab1_rate"),effective_mab1_rate.rename("mab1_rate"),],axis=1,)

        # Get a random draw from randomness system for each simulant
        draw = self.randomness.get_draw(healthy_index)

        # Align draw with the merged DF
        draw, merged_df = draw.align(merged_df)

        # Add draw to merged DF
        merged_df = pd.concat([merged_df, draw.rename("draw")], axis=1)

        # Initialize future_state column
        merged_df["future_state"] = ""

        # Determine future state based on the draw and transition rates
        merged_df.loc[merged_df["draw"] < merged_df["ab1_rate"], "future_state"] = "Ab1"
        merged_df.loc[(merged_df["draw"] >= merged_df["ab1_rate"]) & (merged_df["draw"] < (merged_df["ab1_rate"] + merged_df["mab1_rate"])),"future_state"] = "mAb1"

        # Extract the series for simulants transitioning to Ab1 and mAb1 states
        ab1_series = merged_df.loc[merged_df["future_state"] == "Ab1", "future_state"]
        mab1_series = merged_df.loc[merged_df["future_state"] == "mAb1", "future_state"]

        return ab1_series, mab1_series

    def determine_autoantibody(self, event: Event):
        """
        Determines the autoantibody status of individuals in the population.
        Args:
            event (Event): add better description.
        Returns:
            None -> it simply updates state_table in Vivarium
        """
        self.completed_cycles += 1

        if self.completed_cycles < 2:
            return
        # get population
        full_population = self.population_view.get(event.index)
        healthy_index = self._get_healthy_population(full_population)
        # call ab1_rate and mab1rate methods to compute transition probabilities
        effective_ab1_rate = self.ab1_rate(healthy_index)
        effective_mab1_rate = self.mab1_rate(healthy_index)
        # compute future state
        ab1_series, mab1_series = self._compute_future_state(effective_ab1_rate, effective_mab1_rate, healthy_index)
        # update
        self.population_view.update(ab1_series.rename("state"))
        self.population_view.update(mab1_series.rename("state"))

    def determine_ever_antibody(self, event: Event):
        """
        Updates the 'ever_antibody' status for individuals in the population who have ever been in the 'Ab1' or 'mAb1' state.
        Args:
            event (Event): The event containing the index of the population to be evaluated.
        Returns:
            None
        """
        population = self.population_view.get(event.index)
        ab1_mab1_population = population[(population["state"] == "Ab1") | (population["state"] == "mAb1")]
        self.population_view.update(pd.Series(1, index=ab1_mab1_population.index, name="ever_antibody"))

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
