import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vivarium.interface import InteractiveContext
from simulation_package.make_population import Population
from simulation_package.autoantibody import AutoAntibody
from simulation_package.ab1_to_healthy import Ab1ToHealthy
from simulation_package.ab1_to_mab1 import AutoToMultiInsideAutoAntibody
from simulation_package.mab1_to_ab1 import MultiToAutoInsideAutoantibody
from simulation_package.dysglycemia import Dysglycemia 
from simulation_package.from_dysglycemia import FromDysglycemia

# ----------- timer -------------
start_time = time.time()



config = {
    'randomness': {
        'key_columns': ["entrance_time", "GRS2", "fdr"], 
        'random_seed': 1,
    },
    'population': {
        'population_size': 100_000
    },
    'time': {
        'step_size': 365,
    }
}

sim = InteractiveContext(components=[Population(), AutoAntibody(), Ab1ToHealthy(),
                                    AutoToMultiInsideAutoAntibody(),
                                    MultiToAutoInsideAutoantibody(),
                                    Dysglycemia(),
                                    FromDysglycemia()
                                    ], configuration=config)

# Run the simulat

sim.take_steps(2)
state_table = sim.get_population()
print(state_table.head(5))
print(state_table["state"].value_counts())