import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from vivarium.interface import InteractiveContext
from simulation_package.make_population import Population
from simulation_package.observer import StateTableObserver
from simulation_package.autoantibody import AutoAntibody
from simulation_package.dysglycemia import Dysglycemia
from simulation_package.from_dysglycemia import FromDysglycemia
from simulation_package.ab1_to_mab1 import AutoToMultiInsideAutoAntibody
from simulation_package.mab1_to_ab1 import MultiToAutoInsideAutoantibody
from simulation_package.type1_diabetes_dka_splitting import Type1DiabetesDkaSplitting
from simulation_package.ab1_to_healthy import Ab1ToHealthy
from simulation_package.screening_intervention import ScreeningIntervention
from simulation_package.screening import Screening
from simulation_package.objective_function_costs import ObjectiveFunctionCosts
from simulation_package.objective_function_dka import ObjectiveFunctionDKA
#
from simulation_package.uncertain_archiver import Solution, UncertainSol, MeanPerformanceSol, UncertainObjectivesArchiver, UncertainTester

import pymoo
#from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem
import numpy as np



class NoisyProblem(ElementwiseProblem):
    total_simulations = 0

    def __init__(self, **kwargs):

        
        # calls __init__ method of the super_class Problem so that standard pymoo attributes are initialized

        #super().__init__(n_var=10, n_obj=3, xl = np.zeros(10), xu = np.ones(10), **kwargs)
        super().__init__(n_var=15, n_obj=3, xl = np.zeros(15), xu = np.ones(15), **kwargs)


    def _evaluate(self, x, out, *args, **kargs):
   

        objective_values_costs = [] 
        objective_values_dka = []
        objective_non_zero = []
        simulation_counter = 0

        seed = None
        if seed is None:
            seed = np.random.randint(1,2**32-1)

        config = {
            "randomness": {
                "key_columns": ["entrance_time", "GRS2", "fdr"],
                "random_seed": seed,
            },
            "population": {"population_size": 100_000},
            "time": {
                "step_size": 365,
            },
        }
        # run a simulation for each candidate solution 
        screening_vector = x
            
        

        sim = InteractiveContext(components=[Population(), AutoAntibody(), Ab1ToHealthy(),
                                    AutoToMultiInsideAutoAntibody(),
                                    MultiToAutoInsideAutoantibody(),
                                    Dysglycemia(),
                                    FromDysglycemia(),
                                    Screening(continuous_vector= screening_vector),
                                    ScreeningIntervention('screening_intervention', 'further_t1d_splitting_rate'),
                                    Type1DiabetesDkaSplitting(dka_ratio=0.58),
                                    StateTableObserver(),
                                    ObjectiveFunctionCosts(),
                                    ObjectiveFunctionDKA()
                                    ], configuration=config)


        sim.take_steps(len(screening_vector))
        simulation_counter = simulation_counter + 1
        sim.finalize()
        # costs
        obj_function_costs = sim.get_component("objective_function_costs")
        total_costs = obj_function_costs.total_costs
        objective_values_costs.append(total_costs)
        # dka
        obj_function_dka = sim.get_component("objective_function_dka")
        dka_ratio = obj_function_dka.dka_ratio
        objective_values_dka.append(dka_ratio)
    
        #non-zero objective (this is independent of simulation)
        non_zero_counts = sum(1 for value in screening_vector if value > 0)
        objective_non_zero.append(non_zero_counts)

        out["F"] = np.column_stack([objective_values_costs, objective_values_dka, objective_non_zero])
        print(out["F"])
        

    