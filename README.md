


# Development of a Patient-Level Multi-Objective Optimisation Budget Impact Model for Screening Strategies for Childhood Type 1 Diabetes

File structure overview:

Dockerfile                          # add docker?
LICENSE
README.md                          # Project documentation 
background_population_grs.csv      # Background data to compute GRS2 in the population
requirements.txt                   # Python dependencies
simulation_package                 # Core simulation and optimisation modules
├── [disease modules].py           # Modules defining disease state 
├── fire_simulation.py             # Run Vivarium simulation only (no optimisation)
├── run_optimiser_noisy.py         # Run Vivarium simulation + multi-objective optimisation
├── optimiser_noisy.py             # Pymoo optimisation problem definition
└── [support utilities].py         # Additional utilities and observers
tests                              # Unit tests for simulation modules
└── simulation_tests
    └── test_[module].py
transition_probabilities
└── binary_files                   # Survival regression model binaries for transition probabilities
    ├── [state_transition].bin
    └── RiskTable.csv


# Simulation overview

The simulation models individual T1D states through six distinct health states (as in the paper):

    Healthy

    Autoantibody Positive (Ab1)

    Multiple Autoantibodie (mAb1)

    Dysglycemia

    Type 1 Diabetes with/without DKA

Each state is represented by a dedicated Python class with methods that govern:

    Transitions to other states

    Interaction with other states 

    Updating simulant attributes 

Transition probabilities are derived from pre-trained survival regression models stored as .bin files in transition_probabilities/binary_files.


# Multi-Objective Optimisation

This project integrates Pymoo for multi-objective optimisation of screening strategies.

Key files:

    run_optimiser_noisy.py → Runs simulation batches within a Pymoo optimiser loop.

    optimiser_noisy.py → Defines the multi-objective problem (e.g. minimising costs, DKA ratio, number of screenings)


# How to Run Vivarium Simulations

```
python simulation_package/fire_simulation.py
```

This will execute a full microsimulation 

# Run Vivarium Simulations + Multi-Objective Optimisation

```
python simulation_package/fire_simulation.py
```



