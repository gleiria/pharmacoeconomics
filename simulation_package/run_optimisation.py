import time
import multiprocessing
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np 
import pymoo
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import StarmapParallelization
from pymoo.operators.mutation.pm import PolynomialMutation
from simulation_package.optimiser_noisy import NoisyProblem
from simulation_package.custom_mutation import CustomMutation, CombinedMutation
#from simulation_package.optimization_call_back import MyCallback




# ----------- timer -------------
start_time = time.time()

def generate_diverse_population(n_individuals=100, n_genes=15):

    X = np.random.rand(n_individuals, n_genes)

    # Set the first individual to all 1s
    X[0, :] = 1

    remaining_individuals = n_individuals - 1
    screening_spectrum = np.arange(1, n_genes)  

    n_per_group = remaining_individuals // len(screening_spectrum)
    extra = remaining_individuals % len(screening_spectrum)

    individual_idx = 1

    for num_zeros in screening_spectrum:
        n_in_group = n_per_group + (1 if extra > 0 else 0)
        if extra > 0:
            extra -= 1

        for _ in range(n_in_group):
            permuted_indices = np.random.permutation(n_genes)
            zero_positions = permuted_indices[:num_zeros]
            X[individual_idx, zero_positions] = 0
            individual_idx += 1

    return X



# here it goes
screening_problem = NoisyProblem()

if __name__ == "__main__":
    n_processes = 6
    pool = multiprocessing.Pool(n_processes)
    runner = StarmapParallelization(pool.starmap)

    screening_problem = NoisyProblem(elementwise_runner=runner)
      
    X = generate_diverse_population(n_individuals=10, n_genes=15)

    # mutations
    default_mutation = PolynomialMutation(prob=0.1, eta=20) # costumize 
    custom_mutation = CustomMutation(prob=0.3)
    composite_mutation = CombinedMutation(custom_mutation, default_mutation)

    algo_test = NSGA2(pop_size=10, sampling=X, mutation=composite_mutation)

    num_generations = ("n_gen", 5) #250



    results = minimize(problem = screening_problem, algorithm = algo_test, termination = num_generations, save_history = True)


    with open("local_test.pkl", "wb") as file:
        res = pickle.dump(results, file)

print("**************************************************************")
print("--------- executed in %s seconds ---------" % (time.time() - start_time))
print("**************************************************************")




