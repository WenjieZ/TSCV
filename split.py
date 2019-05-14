import warnings
import numbers
from math import ceil, floor, modf
from abc import ABCMeta, abstractmethod
from itertools import chain
from inspect import signature

import numpy as np
from sklearn.utils import indexable, safe_indexing
from sklearn.utils.validation import _num_samples
from sklearn.base import _pprint

__all__ = ['GapCrossValidator',
           'GapLeavePOut',
           'Gapold',
           'GapWalkForward',
           'gap_train_test_split']


SINGLETON_WARNING = "Too few samples. Some training set is a singleton."


def _build_repr(self):
    # XXX This is copied from BaseEstimator's get_params
    cls = self.__class__
    init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
    # Ignore varargs, kw and default values and pop self
    init_signature = signature(init)
    # Consider the constructor parameters excluding 'self'
    if init is object.__init__:
        args = []
    else:
        args = sorted([p.name for p in init_signature.parameters.values()
                       if p.name != 'self' and p.kind != p.VAR_KEYWORD])
    class_name = self.__class__.__name__
    params = dict()
    for key in args:
        # We need deprecation warnings to always be on in order to
        # catch deprecated param values.
        # This is set in utils/__init__.py but it gets overwritten
        # when running under python3 somehow.
        warnings.simplefilter("always", DeprecationWarning)
        try:
            with warnings.catch_warnings(record=True) as w:
                value = getattr(self, key, None)
            if len(w) and w[0].category == DeprecationWarning:
                # if the parameter is deprecated, don't show it
                continue
        finally:
            warnings.filters.pop(0)
        params[key] = value

    return '%s(%s)' % (class_name, _pprint(params, offset=len(class_name)))


class GapCrossValidator(metaclass=ABCMeta):
    """Base class for all gap cross-validators"""

    def __init__(self, gap_before=0, gap_after=0):
        self.gap_before = gap_before
        self.gap_after = gap_after

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, of length n_samples
            The target variable for supervised learning problems.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        for train_index, test_index in zip(
                self._iter_train_indices(X, y, groups),
                self._iter_test_indices(X, y, groups)):
            yield train_index, test_index

    # Since subclasses implement any of the following 4 methods,
    # none can be abstract.
    def _iter_train_indices(self, X=None, y=None, groups=None):
        """Generates integer indices corresponding to training sets.

        By default, delegates to _iter_test_indices(X, y, groups)
        """
        return self.__complement_indices(
                self._iter_test_indices(X, y, groups), _num_samples(X))

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Generates integer indices corresponding to test sets.

        By default, delegates to _iter_test_masks(X, y, groups)
        """
        return GapCrossValidator.__masks_to_indices(
                                self._iter_test_masks(X, y, groups))

    def _iter_test_masks(self, X=None, y=None, groups=None):
        """Generates boolean masks corresponding to test sets.

        By default, delegates to _iter_train_masks(X, y, groups)
        """
        return self.__complement_masks(self._iter_train_masks(X, y, groups))

    def _iter_train_masks(self, X=None, y=None, groups=None):
        """Generates boolean masks corresponding to training sets.

        By default, delegates to _iter_train_indices(X, y, groups)
        """
        return GapCrossValidator.__indices_to_masks(
                self._iter_train_indices(X, y, groups), _num_samples(X))

    @staticmethod
    def __masks_to_indices(masks):
        for mask in masks:
            index = np.arange(len(mask))
            yield index[np.nonzero(mask)]

    @staticmethod
    def __indices_to_masks(indices, n_samples):
        for index in indices:
            mask = np.zeros(n_samples, dtype=np.bool)
            mask[index] = True
            yield mask

    def __complement_masks(self, masks):
        before, after = self.gap_before, self.gap_after
        for mask in masks:
            complement = np.ones(len(mask), dtype=np.bool)
            for i, masked in enumerate(mask):
                if masked:   # then make its neighbourhood False
                    begin = max(i - before, 0)
                    end = min(i + after + 1, len(complement))
                    complement[np.arange(begin, end)] = False
            yield complement

    def __complement_indices(self, indices, n_samples):
        before, after = self.gap_before, self.gap_after
        for index in indices:
            complement = np.arange(n_samples)
            for i in index:
                begin = max(i - before, 0)
                end = min(i + after + 1, n_samples)
                complement = np.setdiff1d(complement, np.arange(begin, end))
            yield complement

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""

    def __repr__(self):
        return _build_repr(self)


class GapLeavePOut(GapCrossValidator):
    """Leave-P-Out cross-validator with Gap

    Provides train/test indices to split data in train/test sets. This results
    in testing on only contiguous samples of size p, while the remaining
    samples (with the gap removed) form the training set in each iteration.

    Parameters
    ----------
    p : int
        Size of the test sets.

    gap_before : int
        Gap before the test sets.

    gap_after : int
        Gap after the test sets.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GapLeavePOut
    >>> glpo = GapLeavePOut(2, 1, 1)
    >>> glpo.get_n_splits([0, 1, 2, 3, 4])
    4
    >>> print(glpo)
    GapLeavePOut(gap_after=1, gap_before=1, p=2)
    >>> for train_index, test_index in glpo.split([0, 1, 2, 3, 4]):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    TRAIN: [3 4] TEST: [0 1]
    TRAIN: [4] TEST: [1 2]
    TRAIN: [0] TEST: [2 3]
    TRAIN: [0 1] TEST: [3 4]
    """

    def __init__(self, p, gap_before=0, gap_after=0):
        super().__init__(gap_before, gap_after)
        self.p = p

    def _iter_test_indices(self, X, y=None, groups=None):
        self.__check_validity(X, y, groups)
        n_samples = _num_samples(X)
        gap_before, gap_after = self.gap_before, self.gap_after
        if n_samples - gap_after - self.p >= gap_before + 1:
            for i in range(n_samples - self.p + 1):
                yield np.arange(i, i + self.p)
        else:
            for i in range(n_samples - gap_after - self.p):
                yield np.arange(i, i + self.p)
            for i in range(gap_before + 1, n_samples - self.p + 1):
                yield np.arange(i, i + self.p)

    def get_n_splits(self, X, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.
        """
        self.__check_validity(X, y, groups)
        n_samples = _num_samples(X)
        gap_before, gap_after = self.gap_before, self.gap_after
        if n_samples - gap_after - self.p >= gap_before + 1:
            n_splits = n_samples - self.p + 1
        else:
            n_splits = max(n_samples - gap_after - self.p, 0)
            n_splits += max(n_samples - self.p - gap_before, 0)
        return n_splits

    def __check_validity(self, X, y=None, groups=None):
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        n_samples = _num_samples(X)
        gap_before, gap_after = self.gap_before, self.gap_after
        if (0 >= n_samples - gap_after - self.p and
                gap_before >= n_samples - self.p):
            raise ValueError("Not enough training samples available.")
        if n_samples - gap_after - self.p <= gap_before + 1:
            warnings.warn(SINGLETON_WARNING, Warning)


class GapKFold(GapCrossValidator):
    """K-Folds cross-validator with Gap

    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds (without shuffling).

    Each fold is then used once as a validation while the k - 1 remaining
    folds (with the gap removed) form the training set.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    gap_before : int
        Gap before the test sets.

    gap_after : int
        Gap after the test sets.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GapKFold
    >>> kf = GapKFold(n_splits=5, gap_before=3, gap_after=4)
    >>> kf.get_n_splits(np.arange(10))
    5
    >>> print(kf)
    GapKFold(gap_after=4, gap_before=3, n_splits=5)
    >>> for train_index, test_index in kf.split(np.arange(10)):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    TRAIN: [6 7 8 9] TEST: [0 1]
    TRAIN: [8 9] TEST: [2 3]
    TRAIN: [0] TEST: [4 5]
    TRAIN: [0 1 2] TEST: [6 7]
    TRAIN: [0 1 2 3 4] TEST: [8 9]

    Notes
    -----
    The first ``n_samples % n_splits`` folds have size
    ``n_samples // n_splits + 1``, other folds have size
    ``n_samples // n_splits``, where ``n_samples`` is the number of samples.
    """

    def __init__(self, n_splits=5, gap_before=0, gap_after=0):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError('The number of folds must be of Integral type. '
                             '%s of type %s was passed.'
                             % (n_splits, type(n_splits)))
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits))

        super().__init__(gap_before, gap_after)
        self.n_splits = n_splits

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        gap_before, gap_after = self.gap_before, self.gap_after
        if n_splits > n_samples:
            raise ValueError(
                ("Cannot have number of splits n_splits={0} greater"
                 " than the number of samples: n_samples={1}.")
                .format(self.n_splits, n_samples))

        indices = np.arange(n_samples)
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int)
        fold_sizes[:n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            if start - gap_before <= 0 and stop + gap_after >= n_samples:
                raise ValueError("Not enough training samples available")
            yield indices[start:stop]
            current = stop

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


def gap_train_test_split(*arrays, **options):
    """Split arrays or matrices into random train and test subsets (with a gap)

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.

    gap_size : float or int, (default=0)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset between the training and the test set. If int,
        represents the absolute number of the dropped samples.

    test_size : float, int, or None, (default=None)
        If float, should be between 0.0 and 1.0 and equal to
        test / (train + test). If int, represents the absolute number of
        test samples. If None, the value is set to the complement of the
        train size and the gap. If ``train_size`` is also None,
        it will be set to 0.25.

    train_size : float, int, or None, (default=None)
        If float, should be between 0.0 and 1.0 and equal to
        train / (train + test). If int, represents the absolute number of
        train samples. If None, the value is automatically set to
        the complement of the test size and the gap size.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import gap_train_test_split
    >>> X, y = np.arange(10).reshape((5, 2)), range(5)
    >>> X
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])
    >>> list(y)
    [0, 1, 2, 3, 4]

    >>> X_train, X_test, y_train, y_test = gap_train_test_split(
    ...     X, y, test_size=0.33, gap_size=1)
    ...
    >>> X_train
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> y_train
    [0, 1, 2]
    >>> X_test
    array([[8, 9]])
    >>> y_test
    [4]

    >>> gap_train_test_split(list(range(10)), gap_size=0.1)
    [[0, 1, 2, 3, 4, 5, 6], [8, 9]]

    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    test_size = options.pop('test_size', None)
    train_size = options.pop('train_size', None)
    gap_size = options.pop('gap_size', 0)

    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))

    arrays = indexable(*arrays)
    n_samples = _num_samples(arrays[0])

    def size_to_number(size, n):
        b, a = modf(size)
        return int(max(a, round(b * n)))

    n_gap = size_to_number(gap_size, n_samples)
    n_remain = n_samples - n_gap
    if test_size is None and train_size is None:
        test_size = 0.25
    if train_size is None:
        n_test = size_to_number(test_size, n_remain)
        n_train = n_remain - n_test
    elif test_size is None:
        n_train = size_to_number(train_size, n_remain)
        n_test = n_remain - n_train
    else:
        warnings.warn("Only test_size is taken into account.", Warning)
        n_test = size_to_number(test_size, n_remain)
        n_train = n_remain - n_test

    train = np.arange(n_train)
    test = np.arange(n_train + n_gap, n_samples)

    return list(chain.from_iterable((safe_indexing(a, train),
                                     safe_indexing(a, test)) for a in arrays))


class GapWalkForward(GapCrossValidator):
    """
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before.
    This cross-validation object is a variation of K-Fold.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, optional
        Maximum size for a single training set.
    test_size : int, optional
        Number of samples in each test set. Defaults to
        ``n_samples / (n_splits + 1)``.
    gap_size : int, default=0
        Number of samples to exclude from the end of each train set before
        the test set.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import TimeSeriesSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> tscv = TimeSeriesSplit(n_splits=5)
    >>> print(tscv)  # doctest: +NORMALIZE_WHITESPACE
    TimeSeriesSplit(gap_size=0, max_train_size=None, n_splits=5,
                    test_size=None)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [0 1 2] TEST: [3]
    TRAIN: [0 1 2 3] TEST: [4]
    TRAIN: [0 1 2 3 4] TEST: [5]
    >>> # Fix test_size to 2 with 12 samples
    >>> X = np.random.randn(12, 2)
    >>> y = np.random.randint(0, 2, 12)
    >>> tscv = TimeSeriesSplit(n_splits=3, test_size=2)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1 2 3 4 5] TEST: [6 7]
    TRAIN: [0 1 2 3 4 5 6 7] TEST: [8 9]
    TRAIN: [0 1 2 3 4 5 6 7 8 9] TEST: [10 11]
    >>> # Add in a 2 period gap
    >>> tscv = TimeSeriesSplit(n_splits=3, test_size=2, gap_size=2)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1 2 3] TEST: [6 7]
    TRAIN: [0 1 2 3 4 5] TEST: [8 9]
    TRAIN: [0 1 2 3 4 5 6 7] TEST: [10 11]
    Notes
    -----
    The training set has size ``i * n_samples // (n_splits + 1)
    + n_samples % (n_splits + 1)`` in the ``i``th split,
    with a test set of size ``n_samples//(n_splits + 1)`` by default,
    where ``n_samples`` is the number of samples.
    """
    def __init__(self, n_splits=5, max_train_size=None, test_size=None,
                 gap_size=0):
        self.n_splits=n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap_size = gap_size

    def get_n_splits(self):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like, with shape (n_samples,)
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap_size = self.gap_size
        test_size = self.test_size if self.test_size else n_samples // n_folds

        # Make sure we have enough samples for the given split parameters
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        if n_samples - gap_size - (test_size * n_splits) <= 0:
            raise ValueError(
                ("Too many splits ={0} for number of samples"
                 " ={1} with test_size ={2} and gap_size ={3}."
                 "").format(n_splits, n_samples, test_size, gap_size))

        indices = np.arange(n_samples)
        test_starts = range(n_samples - n_splits * test_size,
                            n_samples, test_size)

        for test_start in test_starts:
            train_end = test_start - gap_size
            if self.max_train_size and self.max_train_size < train_end:
                yield (indices[train_end - self.max_train_size:train_end],
                       indices[test_start:test_start + test_size])
            else:
                yield (indices[:train_end],
                       indices[test_start:test_start + test_size])
