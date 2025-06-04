import sys
from pathlib import Path

import pandas.testing as pdt
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from simulation_package.mab1_to_ab1 import MultiToAutoInsideAutoantibody



@pytest.fixture
def mab1_to_ab1():
    return MultiToAutoInsideAutoantibody()

@pytest.fixture
def mock_population():
    mock_population_view = MagicMock()

    mock_population_data = {
    "state": ["mAb1"] * 10,
    "previous_state": ["mAb1"] * 10,
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


def test_base_mab1_to_ab1_transition_rate(mab1_to_ab1, mock_event, mock_population):
    # ----------- arrange -------------- 
    # need a mock population view
    mab1_to_ab1.population_view = mock_population
    
    # ----------- act ----------------
    rate = mab1_to_ab1.base_mab1_to_ab1_transition_rate(mock_event.index)

    # ----------- assert -------------
    expected_rate = pd.Series([0.028291] * 10)
    pdt.assert_series_equal(rate, expected_rate, check_names=False, atol=1e-6, rtol=1e-5)


def test_determine_ab1_to_mab1(mab1_to_ab1, mock_event, mock_population):
    #------------ ARRANGE ---------------
    mab1_to_ab1.completed_cycles = 2
    mab1_to_ab1.ab1_population_view = mock_population
  
    # mock mAb1_rate -> everyone given a transition prob of 1
    mock_mab1_to_ab1_rate = MagicMock()
    mock_mab1_to_ab1_rate.return_value = pd.Series(np.ones(10))
    mab1_to_ab1.mab1_to_ab1_rate = mock_mab1_to_ab1_rate

    # mock randomness
    mock_randomness = MagicMock()
    mock_randomness.get_draw = MagicMock(return_value = pd.Series(np.random.random(size=10)))
    mab1_to_ab1.mab1_to_ab1_randomness = mock_randomness

    # need population view to update
    mock_population_view = MagicMock()
    mock_population_view.get = MagicMock(return_value=mock_population.get(mock_event.index))
    mab1_to_ab1.population_view = mock_population_view

   
    #------------- ACT --------------
    mab1_to_ab1.determine_mab1_to_ab1(mock_event)

    # ------------ ASSERT --------------
    # all individuals have trans prob = 1 therefore should all transition 
    expected_series = pd.Series("Ab1", index=mock_event.index, name="state")
    actual_series = mock_population_view.update.call_args_list[0][0][0]
    # assertion
    assert actual_series.equals(expected_series), "Transition logic mAb1 to Ab1 NOT correctly computed"

def test_determine_time_in_state(mab1_to_ab1, mock_population, mock_event):
    #### ARRANGE #####
    mock_population_data = {
        "state": ["mAb1"] * 10,
        "previous_state": ["Ab1", "mAb1"] * 5,
        "time_in_state": [1] * 10
    }
    mock_population_df = pd.DataFrame(mock_population_data)



    # we need a population view to be updated
    mock_population_view = MagicMock()
    mock_population_view.get = MagicMock(return_value=mock_population_df)
    mab1_to_ab1.population_view = mock_population_view
    

    #### ACT ######
    mab1_to_ab1.determine_time_in_state(mock_event)

    #### ASSERT #######
    # assert that all individuals who transitioned have their time in state set to zero
    expected_data = {
        "state": ["mAb1", "mAb1"] * 5,
        "previous_state": ["mAb1", "mAb1"] * 5,
        "time_in_state": [0, 1] * 5,}
    expected_df = pd.DataFrame(expected_data)

    actual_df = mock_population_view.update.call_args_list[0][0][0]
    pd.testing.assert_frame_equal(actual_df, expected_df)

