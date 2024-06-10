import numbers
import warnings
import numpy as np
from scipy import sparse
from sklearn.utils import (
    _safe_indexing,
    check_random_state,
)

from  imblearn.utils import Substitution, check_neighbors_object
from  imblearn.utils._docstring import _n_jobs_docstring, _random_state_docstring
from  imblearn.utils._param_validation import HasMethods, Interval

from imblearn.over_sampling.base import BaseOverSampler

random_seed = 42
np.random.seed(random_seed)

class BaseSMOTE(BaseOverSampler):
    """Base class for the different SMOTE algorithms."""

    _parameter_constraints: dict = {
        **BaseOverSampler._parameter_constraints,
        "k_neighbors": [
            Interval(numbers.Integral, 1, None, closed="left"),
            HasMethods(["kneighbors", "kneighbors_graph"]),
        ],
        "n_jobs": [numbers.Integral, None],
    }

    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Check the NN estimators shared across the different SMOTE
        algorithms.
        """
        self.nn_k_ = check_neighbors_object(
            "k_neighbors", self.k_neighbors, additional_neighbor=1
        )

    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0
    ):
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)

        # np.newaxis for backwards compatability with random_state
        steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])

        X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new

    def _generate_samples(self, X, nn_data, nn_num, rows, cols, steps):
        diffs = nn_data[nn_num[rows, cols]] - X[rows]

        if sparse.issparse(X):
            sparse_func = type(X).__name__
            steps = getattr(sparse, sparse_func)(steps)
            X_sparse = getattr(sparse, sparse_func)(X)
            random = np.random.uniform(0, 1, size=steps.shape)
            X_new1 = X_sparse[rows] + steps.multiply(diffs)
            condition = X_new1 >= nn_data[nn_num[rows, cols]] + 0.5.multiply(diffs)
            X_new = X_sparse[rows] + steps.multiply(diffs) - random.multiply(diffs) if condition.all() else X_sparse[rows] + steps.multiply(diffs) + random.multiply(diffs)
        else:
            random = np.random.uniform(0, 1, size=steps.shape)
            X_new2 = X[rows] + steps * diffs
            condition = X_new2 >= nn_data[nn_num[rows, cols]] + 0.5 * diffs
            X_new = X[rows] + steps*diffs - random*diffs if condition.all() else X[rows] + steps*diffs + random*diffs

        return X_new.astype(X.dtype)

    def _in_danger_noise(self, nn_estimator, samples, target_class, y, kind="danger"):
        x = nn_estimator.kneighbors(samples, return_distance=False)[:, 1:]
        nn_label = (y[x] != target_class).astype(int)
        n_maj = np.sum(nn_label, axis=1)

        if kind == "danger":
            # Samples are in danger for m/2 <= m' < m
            return np.bitwise_and(
                n_maj >= (nn_estimator.n_neighbors - 1) / 2,
                n_maj < nn_estimator.n_neighbors - 1,
            )
        else:  # kind == "noise":
            # Samples are noise for m = m'
            return n_maj == nn_estimator.n_neighbors - 1


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class ISMOTE(BaseSMOTE):
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )

    def _fit_resample(self, X, y):
        # FIXME: to be removed in 0.12
        if self.n_jobs is not None:
            warnings.warn(
                FutureWarning,
            )

        self._validate_estimator()

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X, target_class_indices)

            self.nn_k_.fit(X_class)
            nns = self.nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]
            X_new, y_new = self._make_samples(
                X_class, y.dtype, class_sample, X_class, nns, n_samples, 1.0
            )
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, y_resampled


