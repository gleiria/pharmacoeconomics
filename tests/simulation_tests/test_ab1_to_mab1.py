import sys
from pathlib import Path

import pandas.testing as pdt
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock



sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from simulation_package.ab1_to_mab1 import AutoToMultiInsideAutoAntibody



######################### Fixtures -> arrange steps and data #######################################
@pytest.fixture
def ab1_to_mab1():
    """fixture to mock the real ab1_to_mab1 class"""
    return AutoToMultiInsideAutoAntibody()

@pytest.fixture
def mock_population():
    mock_population_view = MagicMock()

    mock_population_data = {
    "state": ["Ab1"] * 10,
    "previous_state": ["Ab1"] * 10,
    "time_in_state": [2] * 10,
    "GRS2": [15] * 10,
    "fdr": [0] * 10,
    "age": [3] * 10,
    }
    mock_population_df = pd.DataFrame(mock_population_data)
    mock_population_view.get = MagicMock(return_value=mock_population_df)
    return mock_population_view


@pytest.fixture
def mock_event():
    mock_event = MagicMock()
    mock_index = pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    mock_event.index = mock_index
    return mock_event



def test_base_ab1_to_mab1(ab1_to_mab1, mock_event, mock_population):
    #------------ arrange ---------------
    # need a population_view
    ab1_to_mab1.population_view = mock_population

    #act
    rate = ab1_to_mab1.base_ab1_to_mab1_transition_rate(mock_event.index)

    #assert
    expected_rate = pd.Series([0.106955] * 10)
    pdt.assert_series_equal(rate, expected_rate, check_names=False)

def test_determine_ab1_to_mab1(ab1_to_mab1, mock_event, mock_population):
    #------------ ARRANGE ---------------
    ab1_to_mab1.completed_cycles = 2
    ab1_to_mab1.ab1_population_view = mock_population
  
    # mock Ab1_rate -> everyone given a transition prob of 1
    mock_ab1_to_mab1_rate = MagicMock()
    mock_ab1_to_mab1_rate.return_value = pd.Series(np.ones(10))
    ab1_to_mab1.ab1_to_mab1_rate = mock_ab1_to_mab1_rate

    # mock randomness
    mock_randomness = MagicMock()
    mock_randomness.get_draw = MagicMock(return_value = pd.Series(np.random.random(size=10)))
    ab1_to_mab1.ab1_to_mab1_randomness = mock_randomness

    # need population view to update
    mock_population_view = MagicMock()
    mock_population_view.get = MagicMock(return_value=mock_population.get(mock_event.index))
    ab1_to_mab1.population_view = mock_population_view

   
    #------------- ACT --------------
    ab1_to_mab1.determine_ab1_to_mab1(mock_event)

    # ------------ ASSERT --------------
    # all individuals have trans prob = 1 therefore should all transition 
    expected_series = pd.Series("mAb1", index=mock_event.index, name="state")
    actual_series = mock_population_view.update.call_args_list[0][0][0]
    # assertion
    assert actual_series.equals(expected_series), "Transition logic Ab1 to mAb1 NOT correctly computed"


def test_determine_time_in_state(ab1_to_mab1, mock_population, mock_event):
    #### ARRANGE #####
    mock_population_data = {
        "state": ["Ab1"] * 10,
        "previous_state": ["mAb1", "Ab1"] * 5,
        "time_in_state": [1] * 10
    }
    mock_population_df = pd.DataFrame(mock_population_data)



    # we need a population view to be updated
    mock_population_view = MagicMock()
    mock_population_view.get = MagicMock(return_value=mock_population_df)
    ab1_to_mab1.population_view = mock_population_view
    

    #### ACT ######
    ab1_to_mab1.determine_time_in_state(mock_event)

    #### ASSERT #######
    # assert that all individuals who transitioned have their time in state set to zero
    expected_data = {
        "state": ["Ab1", "Ab1"] * 5,
        "previous_state": ["Ab1", "Ab1"] * 5,
        "time_in_state": [0, 1] * 5,}
    expected_df = pd.DataFrame(expected_data)

    actual_df = mock_population_view.update.call_args_list[0][0][0]
    pd.testing.assert_frame_equal(actual_df, expected_df)



