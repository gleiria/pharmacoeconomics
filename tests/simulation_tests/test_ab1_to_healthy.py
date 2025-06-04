import pandas as pd

"""
Module: test_ab1_to_healthy

This module contains unit tests for the Ab1ToHealthy class from the simulation_package.ab1_to_healthy module. 
Tests aim to verify the functionality of transitioning individuals from the Ab1 state to the healthy state.

Fixtures:
    single_to_healthy: Creates an instance of the Ab1ToHealthy class.
    mock_ab1_population: Mocks a population view with individuals in the Ab1 state.
    mock_event: Mocks an event with an index of individuals.

Tests:
    test_base_ab1_to_healthy_rate: Tests the base_ab1_to_healthy_rate method to ensure it computes the correct transition rate.
    test_determine_healthy: Tests the determine_healthy method to ensure individuals transition correctly from Ab1 to healthy state.
"""


import sys
from pathlib import Path

import pandas.testing as pdt
import numpy as np
import pytest
from unittest.mock import MagicMock



sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from simulation_package.ab1_to_healthy import Ab1ToHealthy


######################### Fixtures -> arrange steps and data #######################################
@pytest.fixture
def single_to_healthy():
    """fixture to mock the real class"""
    return Ab1ToHealthy()


@pytest.fixture
def mock_ab1_population():
    """fixture to mock Ab1 population"""
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
    """fixture to mock event object"""
    mock_event = MagicMock()
    mock_index = pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    mock_event.index = mock_index
    return mock_event


def test_base_ab1_to_healthy_rate(single_to_healthy, mock_ab1_population, mock_event):
    """
    Test the base_ab1_to_healthy_rate method.

    Args:
        single_to_healthy: Fixture for the simulation object.
        mock_ab1_population:Fixtrure  mocked AB1 population data.
        mock_event: Fixture Mocked event with an index attribute.
    """
    # arrange using fixtures passed as parameter
    single_to_healthy.ab1_population_view = mock_ab1_population

    # When you fire
    rate = single_to_healthy.base_ab1_to_healthy_rate(mock_event.index)

    # assert that expected rate (taken from survival model) is equal to the rate computed in the method
    expected_rate = pd.Series([0.028689] * 10)
    pdt.assert_series_equal(
        rate, expected_rate, check_names=False, atol=1e-6, rtol=1e-5
    )


def test_determine_healthy(single_to_healthy, mock_ab1_population, mock_event):
    """
    Test the determine_healthy method of the single_to_healthy object.

    Args:
        single_to_healthy: The object being tested.
        mock_ab1_population: Mocked population data for Ab1.
        mock_event: Mocked event data.

    Asserts:
        All individuals transition to healthy state.
    """
    # ------------------- Arrange ------------------- Given

    single_to_healthy.completed_cycles = 2
    single_to_healthy.ab1_population_view = mock_ab1_population

    # mock Ab1_rate -> everyone given a transition prob of 1
    mock_ab1_to_healthy_rate = MagicMock()
    mock_ab1_to_healthy_rate.return_value = pd.Series(np.ones(10))
    single_to_healthy.ab1_to_healthy_rate = mock_ab1_to_healthy_rate

    # need randomness
    mock_randomness = MagicMock()
    mock_randomness.get_draw = MagicMock(
        return_value=pd.Series(np.random.random(size=10))
    )
    single_to_healthy.randomness = mock_randomness

    # need population view to update
    mock_population_view = MagicMock()
    mock_population_view.get = MagicMock(
        return_value=mock_ab1_population.get(mock_event.index)
    )
    single_to_healthy.population_view = mock_population_view

    # ------------------ Act --------------------------- When
    single_to_healthy.determine_healthy(mock_event)

    # ------------------ Assert --------------------------- Then
    # with all individuls with 1 transition prob all of them should transition ot healthy
    expected_ab1_series = pd.Series("healthy", index=mock_event.index, name="state")
    actual_ab1_series = mock_population_view.update.call_args_list[0][0][0]
    # assertion
    assert actual_ab1_series.equals(
        expected_ab1_series
    ), "Transition logic Ab1 to Healthy NOT correctly computed"
