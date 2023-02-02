import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean


def test_everything_works():
    npt.assert_array_equal(np.array([0, 0]), np.array([0, 0]))


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean
    test_array = np.array([[0, 0],
    [0, 0],
    [0, 0]])
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([0, 0]), daily_mean(test_array))

def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean
    test_array = np.array([[1, 2],
    [3, 4],
    [5, 6]])
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([3, 4]), daily_mean(test_array))


@pytest.mark.parametrize(
    "test, expected",
    [
    ([[0, 0], [0, 0], [0, 0]], [0, 0]),
    ([[1, 2], [3, 4], [5, 6]], [3, 4]),
    ])
    
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(np.array(expected), daily_mean(np.array(test)))

@pytest.mark.parametrize(
    "test, expected",
    [
    ([[0, 0], [0, 0], [0, 0]], [0, 0]),
    ([[1, 2], [3, 4], [5, 6]], [1, 2]),
    ])

def test_daily_min(test, expected):
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_min
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array(expected), daily_min(np.array(test)))


@pytest.mark.parametrize(
    "test, expected",
    [
    ([[0, 0], [0, 0], [0, 0]], [0, 0]),
    ([[1, 2], [3, 4], [5, 6]], [5, 6]),
    ])

def test_daily_max(test, expected):
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_max
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array(expected), daily_max(np.array(test)))

def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min
    from pytest import raises
    with raises(TypeError):
        daily_min([['Cannot', 'min'], ['string', 'arguments']])

def test_int_string():
    """Test for TypeError when passing strings"""
    from pytest import raises
    with raises(ValueError):
        int('This will not work')