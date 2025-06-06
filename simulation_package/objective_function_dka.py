import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vivarium import InteractiveContext
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event

class ObjectiveFunctionDKA:

    def __init__(self):
        self.name = "objective_function_dka"
        self.dka_ratio = 0.0


    def setup(self, builder:Builder):
        self.population_view = builder.population.get_view(["state"])
        builder.event.register_listener("simulation_end", self.base_objective_function_dka)

    def base_objective_function_dka(self, event:Event):
        population = self.population_view.get(event.index)
        num_t1d_without_dka = len(population[population["state"] == "T1D_without_DKA"])
        num_t1d_with_dka = len(population[population["state"] == "T1D_with_DKA"])

        self.dka_ratio = num_t1d_with_dka / (num_t1d_with_dka + num_t1d_without_dka)