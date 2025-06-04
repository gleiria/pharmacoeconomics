import sys
from pathlib import Path

import pandas.testing as pdt
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from simulation_package.dysglycemia import Dysglycemia


@pytest.fixture
def dysglycemia():
    return Dysglycemia()

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
def mock_population_mAb1():
    mock_population_view = MagicMock()

    mock_population_data = {
        "state": ["mAb1"] * 10,
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

def test_base_ab1_to_dysglycemia_rate(dysglycemia, mock_event, mock_population):
    # ----------- arrange -------------- 
    # need a mock population view
    dysglycemia.ab1_population_view = mock_population
    
    # ----------- act ----------------
    rate = dysglycemia.base_ab1_to_dysglycemia_rate(mock_event.index)

    # ----------- assert -------------
    expected_rate = pd.Series([1 - 0.920793] * 10)
    pdt.assert_series_equal(rate, expected_rate, check_names=False, atol=1e-6, rtol=1e-5)

def test_base_mab1_to_dysglycemia_rate(dysglycemia, mock_event, mock_population):
    # ----------- arrange -------------- 
    # need a mock population view
    dysglycemia.mab1_population_view = mock_population
    
    # ----------- act ----------------
    rate = dysglycemia.base_mab1_to_dysglycemia_rate(mock_event.index)

    # ----------- assert -------------
    expected_rate = pd.Series([1 - 0.832437] * 10)
    pdt.assert_series_equal(rate, expected_rate, check_names=False, atol=1e-6, rtol=1e-5)

def test_determine_ab1_to_dysglycemia(dysglycemia, mock_event, mock_population):
    #------------ ARRANGE ---------------
    dysglycemia.completed_cycles = 2
    dysglycemia.ab1_population_view = mock_population

    # mock ab1_to_dysglycemia_rate -> everyone given a transition prob of 1
    mock_ab1_to_dysglycemia_rate = MagicMock()
    mock_ab1_to_dysglycemia_rate.return_value = pd.Series(np.ones(10))
    dysglycemia.ab1_to_dysglycemia_rate = mock_ab1_to_dysglycemia_rate
    
    # mock randomness
    mock_randomness = MagicMock()
    mock_randomness.get_draw = MagicMock(return_value = pd.Series(np.random.random(size=10)))
    dysglycemia.randomness_ab1_to_dysglycemia = mock_randomness

    # need population view to update
    mock_population_view = MagicMock()
    mock_population_view.get = MagicMock(return_value=mock_population.get(mock_event.index))    
    dysglycemia.population_view = mock_population_view

    #------------ ACT -------------------
    dysglycemia.determine_ab1_to_dysglycemia(mock_event)

    #------------ ASSERT ----------------   
    # check that the state of the affected individuals is updated to "dysglycemic"
    expected_series = pd.Series("dysglycemic", index=mock_event.index, name="state")
    actual_series = mock_population_view.update.call_args_list[0][0][0]
    #assertion
    assert actual_series.equals(expected_series), "Transition logic Ab1 to dysglycemic NOT correctly computed"

def test_determine_mab1_to_dysglycemia(dysglycemia, mock_event, mock_population):
    #------------ ARRANGE ---------------
    dysglycemia.completed_cycles = 2
    dysglycemia.mab1_population_view = mock_population

    # mock mab1_to_dysglycemia_rate -> everyone given a transition prob of 1
    mock_mab1_to_dysglycemia_rate = MagicMock()
    mock_mab1_to_dysglycemia_rate.return_value = pd.Series(np.ones(10))
    dysglycemia.mab1_to_dysglycemia_rate = mock_mab1_to_dysglycemia_rate
    
    # mock randomness
    mock_randomness = MagicMock()
    mock_randomness.get_draw = MagicMock(return_value = pd.Series(np.random.random(size=10)))
    dysglycemia.randomness_mab1_to_dysglycemia = mock_randomness

    # need population view to update
    mock_population_view = MagicMock()
    mock_population_view.get = MagicMock(return_value=mock_population.get(mock_event.index))    
    dysglycemia.population_view = mock_population_view

    #------------ ACT -------------------
    dysglycemia.determine_mab1_to_dysglycemia(mock_event)

    #------------ ASSERT ----------------   
    # check that the state of the affected individuals is updated to "dysglycemic"
    expected_series = pd.Series("dysglycemic", index=mock_event.index, name="state")
    actual_series = mock_population_view.update.call_args_list[0][0][0]
    #assertion
    assert actual_series.equals(expected_series), "Transition logic mAb1 to dysglycemic NOT correctly computed"

def test_determine_time_in_state(dysglycemia, mock_event):
    #------------ ARRANGE ---------------
    mock_population_data = {
        "state": ["dysglycemic"] * 10,
        "previous_state": ["Ab1", "dysglycemic"] * 5,
        "time_in_state": [1] * 10,
    }
    mock_population_df = pd.DataFrame(mock_population_data)
    
    # need a population view to update
    mock_population_view = MagicMock()
    mock_population_view.get = MagicMock(return_value=mock_population_df)
    dysglycemia.population_view = mock_population_view

    #------------ ACT -------------------
    dysglycemia.determine_time_in_state(mock_event)
    #------------ ASSERT ----------------

    # assert that all individuals who transitioned have their time in state set to zero
    expected_data = {
        "state": ["dysglycemic", "dysglycemic"] * 5,
        "previous_state": ["dysglycemic", "dysglycemic"] * 5,
        "time_in_state": [0, 1] * 5,}
    expected_df = pd.DataFrame(expected_data)

    actual_df = mock_population_view.update.call_args_list[0][0][0]
    pd.testing.assert_frame_equal(actual_df, expected_df)

  

   
    
    

