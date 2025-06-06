import pandas as pd
import pandas.testing as pdt
import numpy as np
from django.test import TestCase
import pytest
from unittest.mock import MagicMock

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from simulation_package.observer import StateTableObserver
from vivarium.interface import InteractiveContext



def test_state_table_observer():
    pass