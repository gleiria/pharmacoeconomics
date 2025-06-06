"""
This module contains the Observer component which is a collector of state_tables for
every year (cycle) in the simulation. Collectes state_tables and saves them in a python list as 
pandas dataframes. 
"""

import os
import sys
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import numpy as np

class StateTableObserver:
    """
    Component creates the Observer()
    """

    def __init__(self):
        self.name = "state_table_observer"
        self.state_tables_list = []
        self.population_view = None
        self.completed_cycles = 0

    def setup(self, builder: Builder):
        """
        The setup method gives the component access to an instance of the Builder
        """
        self.population_view = builder.population.get_view(
            [
                "age",
                "GRS2",
                "fdr",
                "state",
                "time_in_state",
                "previous_state",
                "screened_in_past",
                "screen_status",
                "number_of_screens",
                "screening_cost",
                "t1d_cost",
                "market_basket_cost",
                "ever_antibody"
            ]
        )

        builder.event.register_listener("time_step", self.record_state_table)
     
        

    def record_state_table(self, event: Event):
        """
        this method is called at the end of each time step (year) in the simulation.
        It can be used to collect any data from the simulation at that point in time.
        I add all state_tables to list as I used to build some animated plots
        """
        current_state_table = self.population_view.get(event.index)
        self.state_tables_list.append(current_state_table.copy())

      



    


    