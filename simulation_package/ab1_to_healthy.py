"""
This module contains the Ab1ToHealthy class (component) which handles transitions from Ab1 to Healthy state

Methods:
    setup method: automatically called by the simulation egine which passes in a ``builder`` object which provides access to a variety of framework subsystems and metadata.

                registers healthy_from_ab1_rate in engine Value System. Later, in other components, we might register value modifiers that act upon this rate

    base_healthy_from_ab1_rate: this method computes transition probabilites for simulants passed in pd.index


    determine_healthy_from_ab1_rate: handles transition logic and uses transitions computed by the two methods above


Example usage:
    >>> from simulation_package.ab1_to_mab1 import Ab1ToHealthy
    >>> pass it as an argument to InteractiveContext(components)

Note:
    1) This module requires two key simulation engine systems: Builder and which must be imported from vivarium.framework
    2) This modules requires external serialized survival regression models (compute transition probabilities) to be imported, loaded with pickle and used in base_healthy_rate
        
"""

import os
import sys
import pickle
import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

######## GET BINARY FILES ###########

# dysglycemic to type1_diabetes
with open(
    "transition_probabilities/binary_files/sAB2Healthy.bin", "rb"
) as binary_ab1_to_healthy:
    ab1_to_healthy_model = pickle.load(binary_ab1_to_healthy)


class Ab1ToHealthy:
    """Class to handle the transition of individuals from the Ab1 state to a healthy state.
    

        Initialize the Ab1ToHealthy class.
        Set up the necessary views, rate producers, and event listeners for the model.
            builder (Builder): The builder object to set up the model.
            event (Event): Triggering event (time step)."""
    
    def __init__(self):
        self.name = "single_to_healthy"
        self.completed_cycles = 0

    def setup(self, builder: Builder):
        """
        The setup method gives the component access to an instance of the Builder which exposes a handful of tools to help build components.
        The simulation framework is responsible for calling the setup method on components and providing the builder to them

        Args:
            builder (Builder): 
        """
        self.population_view = builder.population.get_view(
            ["state", "previous_state", "time_in_state"]
        )
        self.ab1_population_view = builder.population.get_view(
            ["state", "previous_state", "time_in_state", "GRS2", "fdr", "age"],
            query="state == 'Ab1'",
        )

        self.ab1_to_healthy_rate = builder.value.register_rate_producer(
            "healthy_rate", source=self.base_ab1_to_healthy_rate
        )
        self.randomness = builder.randomness.get_stream("ab1_to_healthy_model")

        builder.event.register_listener("time_step", self.determine_healthy)

    def base_ab1_to_healthy_rate(self, index: pd.Index) -> pd.Series:
        """
        Calculate the base rate of transitioning from AB1 to healthy state.
        Args:
            index (pd.Index): Index of the population data.
        Returns:
            pd.Series: Series representing the rate of transitioning to a healthy state.
        """
        # Get the population data for the given index
        population_df = self.ab1_population_view.get(index)
        # Predict the survival probability using the external model
        survival_prob = ab1_to_healthy_model.predict_survival_function(
            population_df, times=1, conditional_after=population_df["time_in_state"]
        )
        # Calculate the transition rate from AB1 to healthy
        rate = 1 - survival_prob.iloc[0]
        # Return the calculated rate
        return rate

    def determine_healthy(self, event: Event):
        """
        Handle transitions from AB1 to healthy state.
        Args:
            event (Event): triggering event (time step).
        """
        self.completed_cycles += 1

        if self.completed_cycles < 2:
            return

        # Get the current AB1 population
        ab1_population = self.ab1_population_view.get(event.index)
        ab1_index = ab1_population.index

        # Calculate the effective transition rate from AB1 to healthy
        effective_ab1_to_healthy = self.ab1_to_healthy_rate(ab1_index)

        # Draw random numbers for each individual in the AB1 population
        draw_ab1_to_healthy = self.randomness.get_draw(ab1_index)

        # Determine which individuals transition to healthy based on the draw and effective rate
        affected_ab1_to_healthy = draw_ab1_to_healthy < effective_ab1_to_healthy

        # Update the state of individuals who transition to healthy
        self.population_view.update(
            pd.Series("healthy", index=ab1_index[affected_ab1_to_healthy], name="state")
        )
