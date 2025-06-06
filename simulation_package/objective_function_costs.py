import numpy as np
import pandas as pd
from scipy.stats import gamma
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vivarium import InteractiveContext
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event


class ObjectiveFunctionCosts:

    def __init__(self,autoantibody_test_cost = 92, genetic_test_cost = 50, t1d_without_dka_cost = 13880, t1d_with_dka_cost = 23529):
        self.name = "objective_function_costs"
        self.number_genetic_screens = 100000
        self.autoantibody_test_cost = autoantibody_test_cost
        self.genetic_test_cost = genetic_test_cost
        self.t1d_without_dka_cost = t1d_without_dka_cost
        self.t1d_with_dka_cost = t1d_with_dka_cost
        self.total_costs = 0.0

    def setup(self, builder:Builder):
        self.population_view = builder.population.get_view(["state","number_of_screens","market_basket_cost","t1d_cost"])
        builder.event.register_listener("simulation_end", self.base_objective_function)

    def base_objective_function(self, event:Event):
        
        population = self.population_view.get(event.index)
        total_autoantibody_screens = population["number_of_screens"].sum()

        num_t1d_without_dka = len(population[population["state"] == "T1D_without_DKA"])
        num_t1d_with_dka = len(population[population["state"] == "T1D_with_DKA"])
        total_t1d_management_costs =  population["t1d_cost"].sum()

        self.total_costs = (
            total_autoantibody_screens * self.autoantibody_test_cost +
            self.number_genetic_screens * self.genetic_test_cost +
            num_t1d_without_dka * self.t1d_without_dka_cost +
            num_t1d_with_dka * self.t1d_with_dka_cost + total_t1d_management_costs
        ) / 100000 
        
