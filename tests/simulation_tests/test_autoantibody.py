"""This module tests two critical functions in the Autoantibody component:

        -> age_base_ab1_transition_rate()
        -> age_mbase_ab1_transition_rate

        These two functions generate individual transition probabilities for each individual given their:
            -> age
            -> grs
            -> family_history

"""

import pandas as pd
import pandas.testing as pdt
import numpy as np
from django.test import TestCase
import pytest
from unittest.mock import MagicMock

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from simulation_package.autoantibody import AutoAntibody
from vivarium.interface import InteractiveContext


######################### Fixtures -> arrange steps and data #######################################


@pytest.fixture
def autoantibody():
    return AutoAntibody()


@pytest.fixture
def mock_population():
    mock_population_view = MagicMock()
    mock_population_data = {
        "state": ["healthy"] * 10,
        "previous_state": ["healthy"] * 10,
        "time_in_state": [2] * 10,
        "GRS2": [15] * 10,
        "fdr": [0] * 10,
        "market_basket_cost": [0] * 10,
        "ever_antibody": [0] * 10,
    }
    mock_population_df = pd.DataFrame(mock_population_data)
    mock_population_view.get = MagicMock(return_value=mock_population_df)
    return mock_population_view


@pytest.fixture
def mock_event():
    mock_event = MagicMock()
    mock_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    mock_event.index = mock_index
    return mock_event


######################### tests request what they need #######################################


def test_base_ab1_transition_rate(autoantibody, mock_population):
    autoantibody.population_view = mock_population

    # When you fire
    mock_index = pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    rate = autoantibody.base_ab1_transition_rate(mock_index)

    # Then assert that expected Series and rate
    # this is very tricky and don't forget this! These are floating point numbers and automatically rounded
    # when printed to the screen. so in fact the numbers are different and pandas assertions pick that
    # also there are other hidden differences. Do not trust printed output alone. atol and rtol allow for subtil differences
    expected_rate = pd.Series([0.009616] * 10)
    pdt.assert_series_equal(
        rate, expected_rate, check_names=False, atol=1e-6, rtol=1e-5
    )


def test_determine_autoantibody(autoantibody, mock_population, mock_event):
    # --- Given a state of the world where all simulants have 1 probability to transition to Ab1 and 0 prob to transition o mAb1 --- #
    # autoantibody = AutoAntibody()
    autoantibody.completed_cycles = 2
    # mock population view
    mock_population_view = MagicMock()
    # mock_population_view.get = MagicMock(return_value=mock_population.loc[mock_index])
    mock_population_view.get = MagicMock(
        return_value=mock_population.get(mock_event.index)
    )
    autoantibody.population_view = mock_population_view

    # mock Ab1_rate
    mock_Ab1_rate = MagicMock()
    mock_Ab1_rate.return_value = pd.Series(np.ones(10))
    autoantibody.ab1_rate = mock_Ab1_rate

    # mock mAb1_rate
    mock_mAb1_rate = MagicMock()
    mock_mAb1_rate.return_value = pd.Series(np.zeros(10))
    autoantibody.mab1_rate = mock_mAb1_rate

    mock_randomness = MagicMock()
    mock_randomness.get_draw = MagicMock(
        return_value=pd.Series(np.random.random(size=10))
    )
    # mock_randomness.return_value = pd.Series(np.random.random(size=9))
    autoantibody.randomness = mock_randomness

    # --- When determine_autoantibody() method  is called --- #
    autoantibody.determine_autoantibody(mock_event)

    # # --- Then assert that the update() method of population_view is called with correct parameters --- #
    # 1) Ab1
    expected_Ab1_series = pd.Series("Ab1", index=mock_event.index, name="state")
    actual_Ab1_series = mock_population_view.update.call_args_list[0][0][0]
    assert actual_Ab1_series.equals(
        expected_Ab1_series
    ), "Expected Ab1 series not passed as argument to update method()"

    # 2) mAb1
    expected_mAb1_series = pd.Series([], name="state", dtype=object)
    actual_mAb1_series = mock_population_view.update.call_args_list[1][0][0]
    assert actual_mAb1_series.equals(
        expected_mAb1_series
    ), "Expected mAb1 series not passed as argument to update method()"


def test_determine_ever_autoantibody(autoantibody, mock_event):
    mock_population_view = MagicMock()
    # given a population of healthy and Ab1 individuals
    mock_antibody_data = {"state": ["Ab1", "healthy"] * 5}
    mock_antibody_df = pd.DataFrame(mock_antibody_data)

    mock_population_view = MagicMock()
    mock_population_view.get = MagicMock(return_value=mock_antibody_df)
    autoantibody.population_view = mock_population_view

    # when determine_ever_autoantibody fires
    autoantibody.determine_ever_antibody(mock_event)

    # then I want to assert that update method is called with series len(5) ones
    expected_data = np.array([1, 1, 1, 1, 1])
    expected_series = pd.Series(expected_data, index=[0, 2, 4, 6, 8], name="ever_antibody")
    actual_series = mock_population_view.update.call_args_list[0][0][0]
    assert actual_series.equals(
        expected_series
    ), "test_determine_ever_autoantibody update() method not called with correct data."


def test_determine_time_in_state(autoantibody, mock_event):
    mock_population_data = {
        "state": ["Ab1", "healthy"] * 5,
        "previous_state": ["healthy"] * 10,
    }
    mock_population_df = pd.DataFrame(mock_population_data)

    mock_population_view = MagicMock()
    mock_population_view.get = MagicMock(return_value=mock_population_df)
    autoantibody.population_view = mock_population_view
    autoantibody.determine_time_in_state(mock_event)

    expected_data = {
        "state": ["Ab1", "healthy"] * 5,
        "previous_state": ["Ab1", "healthy"] * 5,
        "time_in_state": [0.0, np.nan] * 5,
    }

    expected_df = pd.DataFrame(expected_data)
    actual_df = mock_population_view.update.call_args_list[0][0][0]
    pd.testing.assert_frame_equal(actual_df, expected_df)
