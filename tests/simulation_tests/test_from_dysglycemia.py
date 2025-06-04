import sys
from pathlib import Path

import pandas.testing as pdt
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from simulation_package.from_dysglycemia import FromDysglycemia

@pytest.fixture
def from_dysglycemia():
    return FromDysglycemia()


@pytest.fixture
def mock_population():
    mock_population_view = MagicMock()

    mock_population_data = {
        "state": ["dysglycemic"] * 10,
        "previous_state": ["dysglycemic"] * 10,
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

def test_base_dysglycemia_to_ab1_rate(from_dysglycemia, mock_event, mock_population):
    # ----------- arrange --------------
    # need a mock population view
    from_dysglycemia.dysglycemic_population_view = mock_population

    # --------------- act ----------------
    rate = from_dysglycemia.base_dysglycemia_to_ab1_rate(mock_event.index)

    # --------------- assert --------------
    expected_rate = pd.Series([1-0.951109]*10) 
    pdt.assert_series_equal(rate, expected_rate, check_names=False, atol=1e-6, rtol=1e-5)
    

def test_base_dysglycemia_to_mab1_rate(from_dysglycemia, mock_event, mock_population):
    # ----------- arrange --------------
    from_dysglycemia.dysglycemic_population_view = mock_population

    # --------------- act ----------------
    rate = from_dysglycemia.base_dysglycemia_to_mab1_rate(mock_event.index)

    # --------------- assert --------------
    expected_rate = pd.Series([1-0.888334]*10) 
    pdt.assert_series_equal(rate, expected_rate, check_names=False, atol=1e-6, rtol=1e-5)
    

def test_base_dysglycemia_to_t1d_rate(from_dysglycemia, mock_event, mock_population):
    # ----------- arrange --------------
    from_dysglycemia.dysglycemic_population_view = mock_population

    # --------------- act ----------------
    rate = from_dysglycemia.base_dysglycemia_to_t1d_rate(mock_event.index)
    
    # --------------- assert --------------
    expected_rate = pd.Series([1-0.750805]*10) 
    pdt.assert_series_equal(rate, expected_rate, check_names=False, atol=1e-6, rtol=1e-5)

def test_determine_from_dysglycemia(from_dysglycemia, mock_event, mock_population):
    # ----------- arrange --------------
    from_dysglycemia.completed_cycles = 2
    from_dysglycemia.dysglycemic_population_view = mock_population

    # mock dysglycemia_to_ab1_rate
    mock_dysglycemia_to_ab1_rate = MagicMock()
    mock_dysglycemia_to_ab1_rate.return_value = pd.Series(np.zeros(10))
    from_dysglycemia.dysglycemia_to_ab1_rate = mock_dysglycemia_to_ab1_rate

    # mock dysglycemia_to_mab1_rate
    mock_dysglycemia_to_mab1_rate = MagicMock() 
    mock_dysglycemia_to_mab1_rate.return_value = pd.Series(np.zeros(10))     
    from_dysglycemia.dysglycemia_to_mab1_rate = mock_dysglycemia_to_mab1_rate

    # mock dysglycemia_to_t1d_rate
    mock_dysglycemia_to_t1d_rate = MagicMock()
    mock_dysglycemia_to_t1d_rate.return_value = pd.Series(np.ones(10))
    from_dysglycemia.dysglycemia_to_t1d_rate = mock_dysglycemia_to_t1d_rate

    # mock randomness
    mock_randomness = MagicMock()
    mock_randomness.get_draw = MagicMock(return_value=pd.Series(np.random.random(size=10)))
    from_dysglycemia.randomness = mock_randomness

    # need population view to update
    mock_population_view = MagicMock()
    mock_population_view.get = MagicMock(return_value=mock_population.get(mock_event.index))
    from_dysglycemia.population_view = mock_population_view
    # --------------- act ----------------      
    from_dysglycemia.determine_from_dysglycemia(mock_event)


    # --------------- assert --------------
    expected_ab1_series = pd.Series([], name="state", dtype=object)
    actual_series = mock_population_view.update.call_args_list[0][0][0]
    # assertion 
    assert actual_series.equals(expected_ab1_series), "Transition logic from dysglycemia to Ab1 NOT correctly computed"

        # --------------- assert --------------
    expected_mab1_series = pd.Series([], name="state", dtype=object)
    actual_series = mock_population_view.update.call_args_list[1][0][0]
    # assertion 
    assert actual_series.equals(expected_mab1_series), "Transition logic from dysglycemia to mAb1 NOT correctly computed"

    expected_t1d_series = pd.Series("type1_diabetes", index=mock_event.index, name="state")
    actual_series = mock_population_view.update.call_args_list[2][0][0]
    # assertion 
    assert actual_series.equals(expected_t1d_series), "Transition logic from dysglycemia to type1_diabetes NOT correctly computed"


