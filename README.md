[![Downloads](https://pepy.tech/badge/tscv/month)](https://pepy.tech/project/tscv)
[![Build Status](https://travis-ci.com/WenjieZ/TSCV.svg?branch=master)](https://travis-ci.com/WenjieZ/TSCV)
[![codecov](https://codecov.io/gh/WenjieZ/TSCV/branch/master/graph/badge.svg?token=dcGlEfHCw2)](https://codecov.io/gh/WenjieZ/TSCV)
[![Documentation Status](https://readthedocs.org/projects/tscv/badge/?version=latest)](https://tscv.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/186586661.svg)](https://zenodo.org/badge/latestdoi/186586661)

![](train-gap-test.svg)

# TSCV: Time Series Cross-Validation

This repository is a [scikit-learn](https://scikit-learn.org) extension for time series cross-validation.
It introduces **gaps** between the training set and the test set, which mitigates the temporal dependence of time series and prevents information leakage.

## Installation

```bash
pip install tscv
```

or

```bash
conda install -c conda-forge tscv
```

## Usage

This extension defines 4 cross-validator classes and 1 function:
- `GapLeavePOut`
- `GapKFold`
- `GapRollForward`
- `CombinatorialGapKFold`
- `gap_train_test_split`

The four classes can all be passed, as the `cv` argument, to
scikit-learn functions such as `cross-validate`, `cross_val_score`,
and `cross_val_predict` (except `CombinatorialGapKFold`), just like the native cross-validator classes.

The one function is an alternative to the `train_test_split` function in `scikit-learn`.

## Examples

The following example uses `GapKFold` instead of `KFold` as the cross-validator.
```python
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from tscv import GapKFold

iris = datasets.load_iris()
clf = svm.SVC(kernel='linear', C=1)

# use GapKFold as the cross-validator
cv = GapKFold(n_splits=5, gap_before=5, gap_after=5)
scores = cross_val_score(clf, iris.data, iris.target, cv=cv)
```

The following example uses `gap_train_test_split` to split the data set into the training set and the test set.
```python
import numpy as np
from tscv import gap_train_test_split

X, y = np.arange(20).reshape((10, 2)), np.arange(10)
X_train, X_test, y_train, y_test = gap_train_test_split(X, y, test_size=2, gap_size=2)
```

## Contributing
- Report bugs in the issue tracker
- Express your use cases in the issue tracker

## Documentations
- [tscv.readthedocs.io](https://tscv.readthedocs.io)

## Acknowledgments

- I would like to thank Jeffrey Racine and Christoph Bergmeir for the helpful discussion.

## License
BSD-3-Clause

## Citation

Wenjie Zheng. (2021). Time Series Cross-Validation (TSCV): an extension for scikit-learn. Zenodo. http://doi.org/10.5281/zenodo.4707309

```latex
@software{zheng_2021_4707309,
  title={{Time Series Cross-Validation (TSCV): an extension for scikit-learn}},
  author={Zheng, Wenjie},
  month={april},
  year={2021},
  publisher={Zenodo},
  doi={10.5281/zenodo.4707309},
  url={http://doi.org/10.5281/zenodo.4707309}
}
```
