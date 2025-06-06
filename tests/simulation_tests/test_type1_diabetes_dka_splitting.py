import sys
from pathlib import Path

import pandas.testing as pdt
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from simulation_package.type1_diabetes_dka_splitting import Type1DiabetesDkaSplitting


@pytest.fixture
def type1_diabetes_dka_splitting():
    return Type1DiabetesDkaSplitting(dka_ratio=0.58)

@pytest.fixture
def mock_population():
    mock_population_view = MagicMock()

    mock_population_data = {
        "state": ["type1_diabetes"] * 10,
        "previous_state": ["type1_diabetes"] * 10,
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

def test_base_further_t1d_splitting_rate(type1_diabetes_dka_splitting, mock_event, mock_population):
    # ----------- arrange --------------
    # need a mock population view
    type1_diabetes_dka_splitting.t1d_population_view = mock_population

    # --------------- act ----------------
    rate = type1_diabetes_dka_splitting.base_further_t1d_splitting_rate(mock_event.index)

    # --------------- assert --------------
    expected_rate = pd.Series([0.58] * 10)
    pdt.assert_series_equal(rate, expected_rate, check_names=False, atol=1e-6, rtol=1e-5)

def test_determine_t1d(type1_diabetes_dka_splitting, mock_event, mock_population):
    # ----------- arrange --------------
    type1_diabetes_dka_splitting.completed_cycles = 2
    type1_diabetes_dka_splitting.t1d_population_view = mock_population

    mock_further_t1d_splitting_rate = MagicMock()
    mock_further_t1d_splitting_rate.return_value = pd.Series([1] * 10, index=mock_event.index)
    type1_diabetes_dka_splitting.further_t1d_splitting_rate = mock_further_t1d_splitting_rate

    mock_randomness = MagicMock()
    mock_randomness.get_draw = MagicMock(return_value=pd.Series(np.random.random(size=10)))
    type1_diabetes_dka_splitting.further_t1d_splitting_randomness = mock_randomness

    mock_population_view = MagicMock()
    mock_population_view.get = MagicMock(return_value=mock_population.get(mock_event.index))
    type1_diabetes_dka_splitting.population_view = mock_population_view
    

    # --------------- act ----------------
    type1_diabetes_dka_splitting.determine_t1d(mock_event)

    # --------------- assert --------------
    expected_t1d_series = pd.Series(["T1D_with_DKA"] * 10, name="state", dtype=object)
    actual_series = mock_population_view.update.call_args_list[0][0][0]
    
    assert actual_series.equals(expected_t1d_series), "Transition logic from type1_diabetes to DKA/No_DKA NOT correctly computed"
