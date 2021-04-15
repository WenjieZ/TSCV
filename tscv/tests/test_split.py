"""Test the split module"""
# import warnings
import pytest
import numpy as np

from numpy.testing import assert_equal
# from numpy.testing import assert_allclose
# from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from sklearn.utils._testing import ignore_warnings

from tscv._split import _build_repr
from tscv import GapCrossValidator
from tscv import GapLeavePOut
from tscv import GapKFold
from tscv import GapWalkForward
from tscv import GapRollForward
from tscv import gap_train_test_split


def test_build_repr():
    class MockSplitter:
        def __init__(self, a, b=0, c=None):
            self.a = a
            self.b = b
            self.c = c

        def __repr__(self):
            return _build_repr(self)

    assert repr(MockSplitter(5, 6)) == "MockSplitter(a=5, b=6, c=None)"


def test_gap_cross_validator():
    # Create a dummy subclass for the test
    class testCV(GapCrossValidator):
        n_splits = 2

        def get_n_splits(self, X=None, y=None, groups=None):
            return testCV.n_splits

        def _iter_train_indices(self, X=None, y=None, groups=None):
            yield [1, 3]
            yield [2, 4]

    cv = testCV(0, 2)
    assert_equal(cv.gap_before, 0)
    assert_equal(cv.gap_after, 2)
    assert_equal(cv.get_n_splits(), 2)

    indices = cv._GapCrossValidator__masks_to_indices(
        [[False, True, True, False, True], [False, False], [True]])
    assert_array_equal(next(indices), [1, 2, 4])
    assert_array_equal(next(indices), [])
    assert_array_equal(next(indices), [0])

    masks = cv._GapCrossValidator__indices_to_masks([[1, 2, 3], [5]], 6)
    assert_array_equal(next(masks), [False, True, True, True, False, False])
    assert_array_equal(next(masks), [False, False, False, False, False, True])

    masks = cv._GapCrossValidator__complement_masks(
        [[False,  True,  True,  True, False, False],
         [False, False, False, False, False,  True]])
    assert_array_equal(next(masks), [True, False, False, False, False, False])
    assert_array_equal(next(masks), [True, True, True, True, True, False])

    indices = cv._GapCrossValidator__complement_indices([[1, 2, 3], [5]], 7)
    assert_array_equal(next(indices), [0, 6])
    assert_array_equal(next(indices), [0, 1, 2, 3, 4])

    masks = cv._iter_train_masks("abcde")
    assert_array_equal(next(masks), [False, True, False, True, False])
    assert_array_equal(next(masks), [False, False, True, False, True])

    masks = cv._iter_test_masks("abcde")
    assert_array_equal(next(masks), [True, False, False, False, False])
    assert_array_equal(next(masks), [True, True, False, False, False])

    indices = cv._iter_test_indices("abcde")
    assert_array_equal(next(indices), [0])
    assert_array_equal(next(indices), [0, 1])

    # Another dummy subclass
    class test2CV(GapCrossValidator):
        n_splits = 2

        def get_n_splits(self, X=None, y=None, groups=None):
            return testCV.n_splits

        def _iter_test_indices(self, X=None, y=None, groups=None):
            yield [1, 3]
            yield [2, 4]

    cv = test2CV()

    indices = cv._iter_train_indices("abcdef")
    assert_array_equal(next(indices), [0, 2, 4, 5])
    assert_array_equal(next(indices), [0, 1, 3, 5])

    splits = cv.split("abcdef")

    train, test = next(splits)
    assert_array_equal(train, [0, 2, 4, 5])
    assert_array_equal(test, [1, 3])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 3, 5])
    assert_array_equal(test, [2, 4])


@ignore_warnings
def test_gap_leave_p_out():
    splits = GapLeavePOut(3, 1, 1).split([0, 1, 2, 3, 4])

    train, test = next(splits)
    assert_array_equal(train, [4])
    assert_array_equal(test, [0, 1, 2])

    train, test = next(splits)
    assert_array_equal(train, [0])
    assert_array_equal(test, [2, 3, 4])

    splits = GapLeavePOut(3, 2, 1).split([0, 1, 2, 3, 4])
    train, test = next(splits)
    assert_array_equal(train, [4])
    assert_array_equal(test, [0, 1, 2])

    splits = GapLeavePOut(2, 1, 1).split([0, 1, 2, 3, 4])

    train, test = next(splits)
    assert_array_equal(train, [3, 4])
    assert_array_equal(test, [0, 1])

    train, test = next(splits)
    assert_array_equal(train, [4])
    assert_array_equal(test, [1, 2])

    train, test = next(splits)
    assert_array_equal(train, [0])
    assert_array_equal(test, [2, 3])

    train, test = next(splits)
    assert_array_equal(train, [0, 1])
    assert_array_equal(test, [3, 4])

    splits = GapLeavePOut(2).split([0, 1, 2, 3, 4])

    train, test = next(splits)
    assert_array_equal(train, [2, 3, 4])
    assert_array_equal(test, [0, 1])

    train, test = next(splits)
    assert_array_equal(train, [0, 3, 4])
    assert_array_equal(test, [1, 2])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 4])
    assert_array_equal(test, [2, 3])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2])
    assert_array_equal(test, [3, 4])

    assert_equal(GapLeavePOut(3, 1, 1).get_n_splits("abcde"), 2)
    assert_equal(GapLeavePOut(3, 2, 1).get_n_splits("abcde"), 1)
    assert_equal(GapLeavePOut(2, 1, 1).get_n_splits("abcde"), 4)
    assert_equal(GapLeavePOut(2).get_n_splits("abcde"), 4)


def test_gap_k_fold():
    splits = GapKFold().split(np.arange(10))

    train, test = next(splits)
    assert_array_equal(train, [2, 3, 4, 5, 6, 7, 8, 9])
    assert_array_equal(test, [0, 1])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 4, 5, 6, 7, 8, 9])
    assert_array_equal(test, [2, 3])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3, 6, 7, 8, 9])
    assert_array_equal(test, [4, 5])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3, 4, 5, 8, 9])
    assert_array_equal(test, [6, 7])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3, 4, 5, 6, 7])
    assert_array_equal(test, [8, 9])

    splits = GapKFold(5, 1, 1).split(np.arange(10))

    train, test = next(splits)
    assert_array_equal(train, [3, 4, 5, 6, 7, 8, 9])
    assert_array_equal(test, [0, 1])

    train, test = next(splits)
    assert_array_equal(train, [0, 5, 6, 7, 8, 9])
    assert_array_equal(test, [2, 3])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 7, 8, 9])
    assert_array_equal(test, [4, 5])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3, 4, 9])
    assert_array_equal(test, [6, 7])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3, 4, 5, 6])
    assert_array_equal(test, [8, 9])

    splits = GapKFold(5, 3, 4).split(np.arange(10))

    train, test = next(splits)
    assert_array_equal(train, [6, 7, 8, 9])
    assert_array_equal(test, [0, 1])

    train, test = next(splits)
    assert_array_equal(train, [8, 9])
    assert_array_equal(test, [2, 3])

    train, test = next(splits)
    assert_array_equal(train, [0])
    assert_array_equal(test, [4, 5])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2])
    assert_array_equal(test, [6, 7])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3, 4])
    assert_array_equal(test, [8, 9])

    assert_equal(GapKFold(5, 3, 4).get_n_splits(np.arange(10)), 5)


@ignore_warnings
def test_gap_train_test_split():
    train, test = gap_train_test_split(np.arange(20))
    assert_equal(len(train), 15)
    assert_array_equal(test, [15, 16, 17, 18, 19])

    train, test = gap_train_test_split(np.arange(10), gap_size=0.1)
    assert_array_equal(train, [0, 1, 2, 3, 4, 5, 6])
    assert_array_equal(test, [8, 9])

    train, test = gap_train_test_split(np.arange(10), gap_size=3)
    assert_array_equal(train, [0, 1, 2, 3, 4])
    assert_array_equal(test, [8, 9])

    train, test = gap_train_test_split(np.arange(10), gap_size=2.1)
    assert_array_equal(train, [0, 1, 2, 3, 4, 5])
    assert_array_equal(test, [8, 9])

    train, test = gap_train_test_split(np.arange(10), gap_size=0.3)
    assert_array_equal(train, [0, 1, 2, 3, 4])
    assert_array_equal(test, [8, 9])

    train, test = gap_train_test_split(np.arange(10), gap_size=2.3)
    assert_array_equal(train, [0, 1, 2, 3, 4])
    assert_array_equal(test, [8, 9])

    train, test = gap_train_test_split(np.arange(10), gap_size=2, test_size=3)
    assert_array_equal(train, [0, 1, 2, 3, 4])
    assert_array_equal(test, [7, 8, 9])

    train, test = gap_train_test_split(np.arange(10), gap_size=2, train_size=3)
    assert_array_equal(train, [0, 1, 2])
    assert_array_equal(test, [5, 6, 7, 8, 9])

    train, test = gap_train_test_split(np.arange(10),
                                       gap_size=2, train_size=3, test_size=4)
    assert_array_equal(train, [0, 1, 2, 3])
    assert_array_equal(test, [6, 7, 8, 9])


def test_walk_forward_cv():
    X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]

    # Should fail if there are more folds than samples
    with pytest.raises(
        ValueError,
        match="Cannot have number of folds.*greater"
    ):
        next(GapWalkForward(n_splits=7).split(X))

    tscv = GapWalkForward(2)

    # Manually check that Time Series CV preserves the data
    # ordering on toy datasets
    splits = tscv.split(X[:-1])
    train, test = next(splits)
    assert_array_equal(train, [0, 1])
    assert_array_equal(test, [2, 3])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3])
    assert_array_equal(test, [4, 5])

    splits = GapWalkForward(2).split(X)

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2])
    assert_array_equal(test, [3, 4])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3, 4])
    assert_array_equal(test, [5, 6])

    # Check get_n_splits returns the correct number of splits
    splits = GapWalkForward(2).split(X)
    n_splits_actual = len(list(splits))
    assert n_splits_actual == tscv.get_n_splits()
    assert n_splits_actual == 2


def _check_walk_forward_max_train_size(splits, check_splits, max_train_size):
    for (train, test), (check_train, check_test) in zip(splits, check_splits):
        assert_array_equal(test, check_test)
        assert len(check_train) <= max_train_size
        suffix_start = max(len(train) - max_train_size, 0)
        assert_array_equal(check_train, train[suffix_start:])


def test_walk_forward_max_train_size():
    X = np.zeros((6, 1))
    splits = GapWalkForward(n_splits=3).split(X)
    check_splits = GapWalkForward(n_splits=3, max_train_size=3).split(X)
    _check_walk_forward_max_train_size(splits, check_splits, max_train_size=3)

    # Test for the case where the size of a fold is greater than max_train_size
    check_splits = GapWalkForward(n_splits=3, max_train_size=2).split(X)
    _check_walk_forward_max_train_size(splits, check_splits, max_train_size=2)

    # Test for the case where the size of each fold is less than max_train_size
    check_splits = GapWalkForward(n_splits=3, max_train_size=5).split(X)
    _check_walk_forward_max_train_size(splits, check_splits, max_train_size=2)


def test_walk_forward_test_size():
    X = np.zeros((10, 1))

    # Test alone
    splits = GapWalkForward(n_splits=3, test_size=3).split(X)

    train, test = next(splits)
    assert_array_equal(train, [0])
    assert_array_equal(test, [1, 2, 3])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3])
    assert_array_equal(test, [4, 5, 6])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3, 4, 5, 6])
    assert_array_equal(test, [7, 8, 9])

    # Test with max_train_size
    splits = GapWalkForward(n_splits=2, test_size=2,
                            max_train_size=4).split(X)

    train, test = next(splits)
    assert_array_equal(train, [2, 3, 4, 5])
    assert_array_equal(test, [6, 7])

    train, test = next(splits)
    assert_array_equal(train, [4, 5, 6, 7])
    assert_array_equal(test, [8, 9])

    # Should fail with not enough data points for configuration
    with pytest.raises(ValueError, match="Too many splits.*with test_size"):
        splits = GapWalkForward(n_splits=6, test_size=2).split(X)
        next(splits)


def test_walk_forward_gap():
    X = np.zeros((10, 1))

    # Test alone
    splits = GapWalkForward(n_splits=2, gap_size=2).split(X)

    train, test = next(splits)
    assert_array_equal(train, [0, 1])
    assert_array_equal(test, [4, 5, 6])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3, 4])
    assert_array_equal(test, [7, 8, 9])

    # Test with max_train_size
    splits = GapWalkForward(n_splits=3, gap_size=2, max_train_size=2).split(X)

    train, test = next(splits)
    assert_array_equal(train, [0, 1])
    assert_array_equal(test, [4, 5])

    train, test = next(splits)
    assert_array_equal(train, [2, 3])
    assert_array_equal(test, [6, 7])

    train, test = next(splits)
    assert_array_equal(train, [4, 5])
    assert_array_equal(test, [8, 9])

    # Test with test_size
    splits = GapWalkForward(n_splits=2, gap_size=2,
                            max_train_size=4, test_size=2).split(X)

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3])
    assert_array_equal(test, [6, 7])

    train, test = next(splits)
    assert_array_equal(train, [2, 3, 4, 5])
    assert_array_equal(test, [8, 9])

    # Test with additional test_size
    splits = GapWalkForward(n_splits=2, gap_size=2, test_size=3).split(X)

    train, test = next(splits)
    assert_array_equal(train, [0, 1])
    assert_array_equal(test, [4, 5, 6])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3, 4])
    assert_array_equal(test, [7, 8, 9])


class TestGapRollForward:
    def test_invalid_input(self):
        X = range(10)

        with pytest.raises(
            ValueError,
            match="No valid splits for the input arguments."
        ):
            next(GapRollForward(min_train_size=3, gap_size=7).split(X))

    def test_default_input(self):
        X = range(3)
        foo = GapRollForward()
        assert_equal(foo.get_n_splits(X), 3)
        splits = foo.split(X)

        train, test = next(splits)
        assert_array_equal(train, [])
        assert_array_equal(test, [0])

        train, test = next(splits)
        assert_array_equal(train, [0])
        assert_array_equal(test, [1])

        train, test = next(splits)
        assert_array_equal(train, [0, 1])
        assert_array_equal(test, [2])

    def test_max_test_size(self):
        X = range(5)
        foo = GapRollForward(max_test_size=2)
        splits = foo.split(X)
        assert_equal(foo.get_n_splits(X), 3)

        train, test = next(splits)
        assert_array_equal(train, [])
        assert_array_equal(test, [0, 1])

        train, test = next(splits)
        assert_array_equal(train, [0, 1])
        assert_array_equal(test, [2, 3])

        train, test = next(splits)
        assert_array_equal(train, [0, 1, 2, 3])
        assert_array_equal(test, [4])

    def test_max_train_size(self):
        X = range(3)
        foo = GapRollForward(max_train_size=1)
        splits = foo.split(X)
        assert_equal(foo.get_n_splits(X), 3)

        train, test = next(splits)
        assert_array_equal(train, [])
        assert_array_equal(test, [0])

        train, test = next(splits)
        assert_array_equal(train, [0])
        assert_array_equal(test, [1])

        train, test = next(splits)
        assert_array_equal(train, [1])
        assert_array_equal(test, [2])

    def test_gap_size(self):
        X = range(7)
        foo = GapRollForward(gap_size=2, max_test_size=2)
        splits = foo.split(X)
        assert_equal(foo.get_n_splits(X), 3)

        train, test = next(splits)
        assert_array_equal(train, [])
        assert_array_equal(test, [2, 3])

        train, test = next(splits)
        assert_array_equal(train, [0, 1])
        assert_array_equal(test, [4, 5])

        train, test = next(splits)
        assert_array_equal(train, [0, 1, 2, 3])
        assert_array_equal(test, [6])

    def test_min_test_size(self):
        X = range(9)
        foo = GapRollForward(gap_size=2, max_test_size=2, min_test_size=2)
        splits = foo.split(X)
        assert_equal(foo.get_n_splits(X), 3)

        train, test = next(splits)
        assert_array_equal(train, [])
        assert_array_equal(test, [2, 3])

        train, test = next(splits)
        assert_array_equal(train, [0, 1])
        assert_array_equal(test, [4, 5])

        train, test = next(splits)
        assert_array_equal(train, [0, 1, 2, 3])
        assert_array_equal(test, [6, 7])

    def test_min_train_size(self):
        X = range(8)
        foo = GapRollForward(gap_size=2, max_test_size=2, min_train_size=2)
        splits = foo.split(X)
        assert_equal(foo.get_n_splits(X), 2)

        train, test = next(splits)
        assert_array_equal(train, [0, 1])
        assert_array_equal(test, [4, 5])

        train, test = next(splits)
        assert_array_equal(train, [0, 1, 2, 3])
        assert_array_equal(test, [6, 7])

    def test_roll_size(self):
        X = range(7)
        foo = GapRollForward(gap_size=2, max_test_size=2, roll_size=1)
        splits = foo.split(X)
        assert_equal(foo.get_n_splits(X), 5)

        train, test = next(splits)
        assert_array_equal(train, [])
        assert_array_equal(test, [2, 3])

        train, test = next(splits)
        assert_array_equal(train, [0])
        assert_array_equal(test, [3, 4])

        train, test = next(splits)
        assert_array_equal(train, [0, 1])
        assert_array_equal(test, [4, 5])

        train, test = next(splits)
        assert_array_equal(train, [0, 1, 2])
        assert_array_equal(test, [5, 6])

        train, test = next(splits)
        assert_array_equal(train, [0, 1, 2, 3])
        assert_array_equal(test, [6])
