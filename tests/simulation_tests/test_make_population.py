"""This module contains tests for the Population() component"""


from unittest.mock import Mock
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from simulation_package.make_population import Population
from vivarium.interface import InteractiveContext




def test_population_initialization_creates_correct_number_of_rows():
    # Given a new Population of simulants and the following configuration
    population = Population()

    config = {
        "population": {
            "population_size": 1_000,
        }
    }

    # When a simulation fires
    sim = InteractiveContext(components=[population], configuration=config)
    state_table = sim.get_population()

    # then assert that has correct row dimensions (simulants)
    assert state_table.shape[0] == 1_000, "Population DataFrame not setup correctly. Number of rows is incorrect."


def test_population_initialization_creates_correct_columns():
    # Given a new Population of simulants and given expected columns to be created and given following configuration
    population = Population()

    expected_columns = {
        "age",
        "GRS2",
        "fdr",
        "state",
        "previous_state",
        "time_in_state",
        "screened_in_past",
        "number_of_screens",
        "screen_status",
        "screening_cost",
        "t1d_cost",
        "market_basket_cost",
        "entrance_time",
        "ever_antibody",
    }

    config = {
        "population": {
            "population_size": 1_000,
        }
    }

    # When a simulation fires
    sim = InteractiveContext(components=[population], configuration=config)
    state_table = sim.get_population()

    # then assert that correct columns were created
    # I use issubset because vivarium internally will create another column(s) hence not using = = 
    assert expected_columns.issubset(set(state_table.columns)), "Population DataFrame not setup correctly. Columns are incorrect."


def test_population_initialization_creates_simulants_at_age_of_zero():
    # Given a new Population of simulants and given the following configuration
    population = Population()

    config = {
        "population": {
            "population_size": 1_000,
        }
    }

    # When a simulation fires
    sim = InteractiveContext(components=[population], configuration=config)
    state_table = sim.get_population()

    # Then check that all simulants initialized at birth (age of zero)
    assert (state_table["age"] == 0).all(), "Population DataFrame not setup correctly. Not all simulants initialized at birth (age of zero)."


def test_population_initialization_creates_simulants_in_healthy_state():
    # Given a new Population of simulants and given the following configuration
    population = Population()

    config = {
        "population": {
            "population_size": 1_000,
        }
    }
    
    # When a simulation fires
    sim = InteractiveContext(components=[population], configuration=config)
    state_table = sim.get_population()
    
    # Then check that all simulants initialized in Healthy state
    assert (state_table["state"] == "healthy").all(), "Population DataFrame not setup correctly. Not all simulants initialized in Healthy state."


def test_population_initialization_creates_simulants_with_GRS2_values_in_range():
    # Given a new Population of simulants and given the following configuration
    population = Population()

    config = {
        "population": {
            "population_size": 1_000,
        }
    }
    
    # When simulation fires
    sim = InteractiveContext(components=[population], configuration=config)
    state_table = sim.get_population()

    # Then check that all simulants initialized with correct GRS2 ranges (0-100)
    assert (0 <= state_table["GRS2"]).all() and (state_table["GRS2"] <= 100).all(), "GRS2 initialization not in range(0, 100)."


def test_population_initialization_creates_simulants_with_FDR_values_correct():
    # Given a new Population of simulants and given the following configuration
    population = Population()

    config = {
        "population": {
            "population_size": 1_000,
        }
    }
    
    # When simulation fires
    sim = InteractiveContext(components=[population], configuration=config)
    state_table = sim.get_population()

    # Then assert that First Degree Relative values are either 0 or 1
    assert set(state_table["fdr"].unique()).issubset({0, 1}), "FDR column not correctly initialized."


# ###################################################### end of first test #############################################




def test_simulants_are_aged_correctly():
    # Given a new Population of simulants and given the following configuration
    pop = Population()
    config = {
        "population": {
            "population_size": 10_000,
        },
        "time": {
            "step_size": 365,
        },
    }

    # When simulation fires for two years
    sim = InteractiveContext(components=[pop], configuration=config)
    sim.take_steps(2)
    state_table = sim.get_population()

    # Then assert that simulants are at the age of 2
    expected_age = 2
    assert (state_table["age"] == expected_age).all(), "Simulants are not being aged correctly."


    def test_grs_randomness_system_initialized_correctly():
        # Given two new Populations and given configuration with fixed seed
        pop_1 = Population()
        pop_2 = Population()

        config = {
            "randomness": {
                "key_columns": ["GRS2"],
                "random_seed": 1,
            },
            "population": {"population_size": 1_000},
        }

        # When two simulations fire
        sim_1 = InteractiveContext(components=[pop_1], configuration=config)
        sim_2 = InteractiveContext(components=[pop_2], configuration=config)
        state_table_1 = sim_1.get_population()
        state_table_2 = sim_2.get_population()

        # Then assert that GRS2 column is exactly the same
        assert state_table_1["GRS2"].equals(state_table_2["GRS2"]), "GRS2 not integrated in Randomness System correctly"

def test_fdr_randomness_system_initialized_correctly():
    # Given two new Populations and given configuration with fixed seed
    pop_1 = Population()
    pop_2 = Population()

    config = {
        "randomness": {
            "key_columns": ["fdr"],
            "random_seed": 1,
        },
        "population": {"population_size": 1_000},
    }

    # When two simulations fire
    sim_1 = InteractiveContext(components=[pop_1], configuration=config)
    sim_2 = InteractiveContext(components=[pop_2], configuration=config)
    state_table_1 = sim_1.get_population()
    state_table_2 = sim_2.get_population()

    # Then assert that fdr (first degree relative) columns are exactly the same
    assert state_table_1["fdr"].equals(state_table_2["fdr"]), "Family History (fdr) not integrated in Randomness System correctly"


    


