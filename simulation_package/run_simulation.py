import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from vivarium.interface import InteractiveContext
from simulation_package.make_population import Population
from simulation_package.autoantibody import AutoAntibody
from simulation_package.ab1_to_healthy import Ab1ToHealthy
from simulation_package.ab1_to_mab1 import AutoToMultiInsideAutoAntibody
from simulation_package.mab1_to_ab1 import MultiToAutoInsideAutoantibody
from simulation_package.dysglycemia import Dysglycemia 
from simulation_package.from_dysglycemia import FromDysglycemia
from simulation_package.type1_diabetes_dka_splitting import Type1DiabetesDkaSplitting
from simulation_package.screening import Screening
from simulation_package.screening_intervention import ScreeningIntervention
from simulation_package.observer import StateTableObserver
from simulation_package.objective_function_costs import ObjectiveFunctionCosts
from simulation_package.objective_function_dka import ObjectiveFunctionDKA

# ----------- timer -------------
start_time = time.time()


dka_list = []
costs = []

for i in range(10):

    config = {
        'randomness': {
            'key_columns': ["entrance_time", "GRS2", "fdr"], 
            'random_seed': i,
        },
        'population': {
            'population_size': 100_000
        },
        'time': {
            'step_size': 365,
        }
    }

    screening_strategy = np.ones(15)

    sim = InteractiveContext(components=[Population(), AutoAntibody(), Ab1ToHealthy(),
                                        AutoToMultiInsideAutoAntibody(),
                                        MultiToAutoInsideAutoantibody(),
                                        Dysglycemia(),
                                        FromDysglycemia(),
                                        Screening(continuous_vector= screening_strategy),
                                        ScreeningIntervention('screening_intervention', 'further_t1d_splitting_rate'),
                                        Type1DiabetesDkaSplitting(dka_ratio=0.58),
                                        StateTableObserver(),
                                        ObjectiveFunctionCosts(),
                                        ObjectiveFunctionDKA()
                                        ], configuration=config)

    # Run the simulation

    sim.take_steps(15)
    sim.finalize()
    state_table = sim.get_population()
    print(state_table.head(5))
    print(state_table["state"].value_counts())
    print("-----------------------------------------------")
    costs = sim.get_component("objective_function_costs")
    total_costs = costs.total_costs
    costs_list.append(total_costs)
    print(total_costs)
    dka = sim.get_component("objective_function_dka")
    dka_ratio = dka.dka_ratio
    dka_list.append(dka_ratio)
    print(dka_ratio)



