Compatibility with Scikit-Learn
===============================

.. currentmodule:: tscv

The classes provided in this package (i.e., :class:`GapLeavePOut`,
:class:`GapKFold`, and :class:`GapRollForward`) are rarely used directly.
More frequently, their instances are sent to a scikit-learn
cross-validator. This page shows this usage.

.. highlight: python

Let us start by preparing the environment::

	import numpy as np
	from sklearn import datasets
	from sklearn import svm
	from sklearn.model_selection import cross_val_score
	from tscv import GapKFold

In this code snippet, :func:`sklearn.model_selection.cross_val_score`
is a cross-validator provided by scikit-learn, and :class:`GapKFold`,
provided by this package, is referred to as *splitter* in this page.

Now let us initiate the Iris dataset and the SVM algorithm::

	iris = datasets.load_iris()
	clf = svm.SVC(kernel='linear', C=1)

.. note:: The Iris dataset is not a time series.
  Here we use it just to showcase the usage of the package.

Now instantiate a splitter::

	cv = GapKFold(n_splits=5, gap_before=5, gap_after=5)

and send it to the cross-validator along with the dataset and
the algorithm::

	scores = cross_val_score(clf, iris.data, iris.target, cv=cv)

``scores`` saves the cross-validation score and is used as the
criterion to evaluate algorithms::

    >>> scores
    array([1.        , 1.        , 0.83333333, 0.96666667, 0.83333333])

Valid Splitters
---------------
* :class:`GapLeavePOut`
* :class:`GapKFold`
* :class:`GapRollForward`

Valid Cross-Validators
----------------------
* :func:`~sklearn.model_selection.cross_validate`
* :func:`~sklearn.model_selection.cross_val_score`
* :func:`~sklearn.model_selection.cross_val_predict`
