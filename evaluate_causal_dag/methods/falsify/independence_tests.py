"""
Conditional independence (CI) test implementations.

Supported methods
-----------------
FISHER_Z
    Fisher's Z transformation of the partial correlation.  Assumes that the
    data are (approximately) multivariate Gaussian and that conditional
    dependence is captured by linear associations.  Fast and reliable for
    continuous data.

MUTUAL_INFO
    Discretisation-based mutual information test.  Suitable for categorical /
    already-discrete variables or when a rough non-parametric check is needed.
    Uses a chi-square approximation on the empirical contingency tables.

KERNEL_BASED
    Kernel-based conditional independence test using HSIC (Hilbert-Schmidt
    Independence Criterion) with an RBF kernel and permutation-based p-value.
    Non-parametric; able to capture non-linear dependencies.  Computational
    complexity is O(n²) in the sample size.  Reference: Gretton et al., 2007;
    Zhang et al., 2011.

APPROX_KERNEL_BASED
    Approximate kernel-based test using Nyström random features and a
    permutation test.  Much faster than :pydata:`KERNEL_BASED` (O(n)) while
    retaining good power.  Reference: Strobl, Zhang & Visweswaran, 2019.

REGRESSION_BASED
    Regression-based test that compares predictive models with and without the
    candidate variable, using Nyström kernel features and an F-test aggregated
    over k-fold cross-validation.  Reference: Chalupka et al., 2018.

GENERALISED_COV_BASED
    Generalised Covariance Measure (GCM).  Computes residuals of non-linear
    regressions of X on Z and Y on Z, then tests whether the product of the
    residuals has zero mean.  Reference: Shah & Peters, 2018.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Collection, FrozenSet, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


class IndependenceTestMethod(str, Enum):
    """Enumeration of available CI test methods."""

    FISHER_Z = "fisher_z"
    MUTUAL_INFO = "mutual_info"
    KERNEL_BASED = "kernel_based"
    APPROX_KERNEL_BASED = "approx_kernel_based"
    REGRESSION_BASED = "regression_based"
    GENERALISED_COV_BASED = "generalised_cov_based"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CITestResult:
    """Result of a single conditional independence test X ⊥ Y | S.

    Attributes
    ----------
    x, y:
        Names of the two variables being tested.
    conditioning_set:
        Frozen set of names of the conditioning variables.
    p_value:
        p-value of the test.
    statistic:
        Test statistic (interpretation depends on *method*).
    is_independent:
        ``True`` when the null hypothesis of independence is *not* rejected at
        the significance level used when the test was run.
    method:
        The method that produced this result.
    """

    x: str
    y: str
    conditioning_set: FrozenSet[str]
    p_value: float
    statistic: float
    is_independent: bool
    method: IndependenceTestMethod

    # Make the conditioning_set always a frozenset even if a list is passed.
    def __post_init__(self) -> None:
        object.__setattr__(self, "conditioning_set", frozenset(self.conditioning_set))

    def __repr__(self) -> str:
        ind = "⊥" if self.is_independent else "⊬"
        cond = f"|{{{', '.join(sorted(self.conditioning_set))}}}" if self.conditioning_set else ""
        return (
            f"CITestResult({self.x} {ind} {self.y}{cond}, "
            f"p={self.p_value:.4f}, method={self.method.value})"
        )


# ---------------------------------------------------------------------------
# Fisher's Z test
# ---------------------------------------------------------------------------


def _partial_correlation(
    data: pd.DataFrame,
    x: str,
    y: str,
    conditioning_set: List[str],
) -> float:
    """Compute the partial correlation of *x* and *y* given *conditioning_set*.

    When the conditioning set is empty the ordinary Pearson correlation is
    returned.
    """
    n = len(data)
    if len(conditioning_set) == 0:
        r, _ = stats.pearsonr(data[x].to_numpy(), data[y].to_numpy())
        return float(r)

    z_mat = np.column_stack(
        [np.ones(n)] + [data[z].to_numpy() for z in conditioning_set]
    )
    x_arr = data[x].to_numpy(dtype=float)
    y_arr = data[y].to_numpy(dtype=float)

    # OLS residuals of x ~ Z and y ~ Z
    def _residuals(arr: np.ndarray) -> np.ndarray:
        coef, _, _, _ = np.linalg.lstsq(z_mat, arr, rcond=None)
        return arr - z_mat @ coef

    r, _ = stats.pearsonr(_residuals(x_arr), _residuals(y_arr))
    return float(r)


def fisher_z_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    conditioning_set: Collection[str],
    alpha: float = 0.05,
) -> CITestResult:
    """Fisher's Z conditional independence test.

    Tests the null hypothesis H₀: X ⊥ Y | *conditioning_set* against the
    alternative of conditional dependence.

    Parameters
    ----------
    data:
        Observed data as a :class:`pandas.DataFrame`.  Columns must include
        *x*, *y*, and all variables in *conditioning_set*.
    x, y:
        Column names of the two variables to test.
    conditioning_set:
        Column names of the variables to condition on.
    alpha:
        Significance level.  The result's ``is_independent`` flag is
        ``True`` when ``p_value > alpha``.

    Returns
    -------
    CITestResult
    """
    cond = list(conditioning_set)
    n = len(data)
    k = len(cond)

    if n - k - 3 <= 0:
        warnings.warn(
            f"Sample size {n} is too small for Fisher's Z test with "
            f"{k} conditioning variables.  Returning p_value=NaN.",
            stacklevel=2,
        )
        return CITestResult(
            x=x,
            y=y,
            conditioning_set=frozenset(cond),
            p_value=float("nan"),
            statistic=float("nan"),
            is_independent=True,
            method=IndependenceTestMethod.FISHER_Z,
        )

    r = _partial_correlation(data, x, y, cond)
    r_clipped = np.clip(r, -(1.0 - 1e-10), 1.0 - 1e-10)
    z_stat = float(np.arctanh(r_clipped) * np.sqrt(n - k - 3))
    p_value = float(2.0 * stats.norm.sf(abs(z_stat)))

    return CITestResult(
        x=x,
        y=y,
        conditioning_set=frozenset(cond),
        p_value=p_value,
        statistic=z_stat,
        is_independent=p_value > alpha,
        method=IndependenceTestMethod.FISHER_Z,
    )


# ---------------------------------------------------------------------------
# Mutual-information / chi-square test (discrete / categorical data)
# ---------------------------------------------------------------------------


def _discretise(series: pd.Series, bins: int = 5) -> pd.Series:
    """Bin a continuous series into *bins* equal-width categories."""
    try:
        return pd.cut(series, bins=bins, labels=False)
    except Exception:
        return series


def mutual_info_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    conditioning_set: Collection[str],
    alpha: float = 0.05,
    bins: int = 5,
) -> CITestResult:
    """Conditional independence test based on (conditional) mutual information.

    The marginal or conditional contingency tables are built from discretised
    values and a chi-square statistic is derived from the observed mutual
    information.  This is a rough test intended for discrete or
    already-binned data; use :func:`fisher_z_test` for continuous variables.

    Parameters
    ----------
    data, x, y, conditioning_set, alpha:
        As in :func:`fisher_z_test`.
    bins:
        Number of equal-width bins used to discretise continuous columns
        before building the contingency table.

    Returns
    -------
    CITestResult
    """
    cond = list(conditioning_set)
    df = data[[x, y] + cond].copy()

    # Discretise all columns
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = _discretise(df[col], bins=bins)

    df = df.dropna()
    n = len(df)

    if len(cond) == 0:
        ct = pd.crosstab(df[x], df[y])
        chi2, p_value, _, _ = stats.chi2_contingency(ct)
        statistic = float(chi2)
    else:
        # Average over strata defined by the conditioning set
        chi2_total = 0.0
        dof_total = 0
        for _, group in df.groupby(cond):
            if len(group) < 2:
                continue
            ct = pd.crosstab(group[x], group[y])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            chi2_val, _, dof, _ = stats.chi2_contingency(ct)
            chi2_total += chi2_val
            dof_total += dof

        statistic = float(chi2_total)
        p_value = (
            float(stats.chi2.sf(chi2_total, dof_total))
            if dof_total > 0
            else 1.0
        )

    return CITestResult(
        x=x,
        y=y,
        conditioning_set=frozenset(cond),
        p_value=p_value,
        statistic=statistic,
        is_independent=p_value > alpha,
        method=IndependenceTestMethod.MUTUAL_INFO,
    )


# ---------------------------------------------------------------------------
# Kernel-based helpers
# ---------------------------------------------------------------------------


def _rbf_kernel_matrix(
    x_arr: np.ndarray,
    y_arr: np.ndarray | None = None,
    precision: float | None = None,
) -> np.ndarray:
    """Compute RBF (Gaussian) kernel matrix with median-heuristic bandwidth.

    Parameters
    ----------
    x_arr:
        Array of shape (n, d).
    y_arr:
        Optional second array.  When ``None``, ``y_arr = x_arr``.
    precision:
        Inverse bandwidth parameter.  When ``None`` the median-heuristic is
        used: ``precision = 1 / (2 * median(||x_i - x_j||²))``.
    """
    if y_arr is None:
        y_arr = x_arr
    x_arr = np.atleast_2d(x_arr)
    y_arr = np.atleast_2d(y_arr)

    # Pairwise squared Euclidean distances
    x_sq = np.sum(x_arr ** 2, axis=1, keepdims=True)
    y_sq = np.sum(y_arr ** 2, axis=1, keepdims=True)
    dist_sq = x_sq + y_sq.T - 2.0 * x_arr @ y_arr.T

    if precision is None:
        med = float(np.median(dist_sq[dist_sq > 0]))
        precision = 1.0 / (2.0 * med) if med > 0 else 1.0

    return np.exp(-precision * dist_sq)


def _centre_kernel(k: np.ndarray) -> np.ndarray:
    """Centre a kernel matrix: H K H, where H = I - 1/n."""
    n = k.shape[0]
    row_mean = k.mean(axis=1, keepdims=True)
    col_mean = k.mean(axis=0, keepdims=True)
    total_mean = k.mean()
    return k - row_mean - col_mean + total_mean


def _hsic_statistic(kx: np.ndarray, ky: np.ndarray) -> float:
    """Compute the unscaled HSIC statistic from centred kernel matrices."""
    n = kx.shape[0]
    return float(np.trace(kx @ ky) / (n * n))


def _residualise_via_ols(
    data: pd.DataFrame,
    targets: list[str],
    conditioning_set: list[str],
) -> np.ndarray:
    """Regress each *target* on *conditioning_set* via OLS and return residuals.

    Returns an (n, len(targets)) array.
    """
    n = len(data)
    z_mat = np.column_stack(
        [np.ones(n)] + [data[c].to_numpy(dtype=float) for c in conditioning_set]
    )
    residuals = []
    for col in targets:
        arr = data[col].to_numpy(dtype=float)
        coef, _, _, _ = np.linalg.lstsq(z_mat, arr, rcond=None)
        residuals.append(arr - z_mat @ coef)
    return np.column_stack(residuals)


# ---------------------------------------------------------------------------
# Kernel-based CI test (HSIC + permutation)
# ---------------------------------------------------------------------------


def kernel_based_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    conditioning_set: Collection[str],
    alpha: float = 0.05,
    n_permutations: int = 200,
    max_num_samples: int = 2000,
) -> CITestResult:
    """Kernel-based conditional independence test (HSIC + permutation).

    Uses the Hilbert-Schmidt Independence Criterion with an RBF kernel.
    A permutation test is used to estimate the p-value.

    For conditional tests (non-empty conditioning set), the effect of the
    conditioning variables is removed via OLS residualisation before
    computing the HSIC statistic.

    Parameters
    ----------
    data, x, y, conditioning_set, alpha:
        As in :func:`fisher_z_test`.
    n_permutations:
        Number of permutations for the permutation test.
    max_num_samples:
        If the dataset has more rows, a random subsample is used to keep
        computational costs manageable (O(n²) kernel computation).

    Returns
    -------
    CITestResult
    """
    cond = list(conditioning_set)
    df = data[[x, y] + cond].dropna()

    # Subsample if too large
    if len(df) > max_num_samples:
        df = df.sample(n=max_num_samples, random_state=42)

    n = len(df)
    if n < 5:
        return CITestResult(
            x=x, y=y, conditioning_set=frozenset(cond),
            p_value=float("nan"), statistic=float("nan"),
            is_independent=True, method=IndependenceTestMethod.KERNEL_BASED,
        )

    if cond:
        residuals = _residualise_via_ols(df, [x, y], cond)
        x_arr = residuals[:, 0:1]
        y_arr = residuals[:, 1:2]
    else:
        x_arr = df[[x]].to_numpy(dtype=float)
        y_arr = df[[y]].to_numpy(dtype=float)

    kx = _centre_kernel(_rbf_kernel_matrix(x_arr))
    ky = _centre_kernel(_rbf_kernel_matrix(y_arr))

    observed = _hsic_statistic(kx, ky)

    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        perm_stat = _hsic_statistic(kx[np.ix_(perm, perm)], ky)
        if perm_stat >= observed:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)

    return CITestResult(
        x=x, y=y, conditioning_set=frozenset(cond),
        p_value=p_value, statistic=observed,
        is_independent=p_value > alpha,
        method=IndependenceTestMethod.KERNEL_BASED,
    )


# ---------------------------------------------------------------------------
# Approximate kernel-based CI test (Nyström features + permutation)
# ---------------------------------------------------------------------------


def approx_kernel_based_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    conditioning_set: Collection[str],
    alpha: float = 0.05,
    n_components: int = 50,
    n_permutations: int = 200,
) -> CITestResult:
    """Approximate kernel-based CI test using Nyström features.

    Approximates the RBF kernel with Nyström random features, then uses a
    permutation test on the cross-covariance of the feature maps.  This is
    much faster than :func:`kernel_based_test` (O(n) vs O(n²)).

    For conditional tests the conditioning variables' effect is removed from
    the approximate features via linear projection (residualisation).

    Parameters
    ----------
    data, x, y, conditioning_set, alpha:
        As in :func:`fisher_z_test`.
    n_components:
        Number of Nyström components per variable.
    n_permutations:
        Number of permutations for the permutation test.

    Returns
    -------
    CITestResult
    """
    cond = list(conditioning_set)
    df = data[[x, y] + cond].dropna()
    n = len(df)

    if n < 5:
        return CITestResult(
            x=x, y=y, conditioning_set=frozenset(cond),
            p_value=float("nan"), statistic=float("nan"),
            is_independent=True,
            method=IndependenceTestMethod.APPROX_KERNEL_BASED,
        )

    n_comp = min(n_components, n - 1)

    def _nystroem_features(arr: np.ndarray) -> np.ndarray:
        arr = np.atleast_2d(arr)
        nc = min(n_comp, arr.shape[0] - 1)
        if nc < 1:
            return arr
        nys = Nystroem(kernel="rbf", n_components=nc, random_state=42)
        return nys.fit_transform(arr)

    x_feat = _nystroem_features(df[[x]].to_numpy(dtype=float))
    y_feat = _nystroem_features(df[[y]].to_numpy(dtype=float))

    if cond:
        z_feat = _nystroem_features(
            df[cond].to_numpy(dtype=float)
        )
        # Residualise: remove Z's linear effect from features
        z_aug = np.column_stack([np.ones(n), z_feat])
        for feat in (x_feat, y_feat):
            coef, _, _, _ = np.linalg.lstsq(z_aug, feat, rcond=None)
            feat -= z_aug @ coef
        # Re-bind names (in-place subtraction already updated arrays, but be explicit)
        x_feat = x_feat
        y_feat = y_feat

    # Test statistic: squared Frobenius norm of cross-covariance
    def _cross_cov_stat(xf: np.ndarray, yf: np.ndarray) -> float:
        cov_mat = (xf.T @ yf) / n
        return float(np.sum(cov_mat ** 2) * n)

    observed = _cross_cov_stat(x_feat, y_feat)

    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        perm_stat = _cross_cov_stat(x_feat[perm], y_feat)
        if perm_stat >= observed:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)

    return CITestResult(
        x=x, y=y, conditioning_set=frozenset(cond),
        p_value=p_value, statistic=observed,
        is_independent=p_value > alpha,
        method=IndependenceTestMethod.APPROX_KERNEL_BASED,
    )


# ---------------------------------------------------------------------------
# Regression-based CI test (Nyström features + F-test + k-fold CV)
# ---------------------------------------------------------------------------


def _f_test_p_value(
    rss_reduced: float,
    rss_full: float,
    df_reduced: int,
    df_full: int,
    n: int,
) -> float:
    """Compute p-value of an F-test comparing a reduced and a full model.

    Returns 1.0 when the F-statistic is non-positive (full model not better).
    """
    if df_full >= n or df_full <= df_reduced:
        return 1.0
    numerator = (rss_reduced - rss_full) / max(df_full - df_reduced, 1)
    denominator = rss_full / max(n - df_full, 1)
    if denominator <= 0 or numerator <= 0:
        return 1.0
    f_stat = numerator / denominator
    return float(stats.f.sf(f_stat, df_full - df_reduced, n - df_full))


def regression_based_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    conditioning_set: Collection[str],
    alpha: float = 0.05,
    max_num_components: int = 40,
    k_folds: int = 3,
) -> CITestResult:
    """Regression-based conditional independence test.

    Tests X ⊥ Y | S by comparing two predictive models for Y:

    * **Reduced model**: Y ~ Nyström(S)  (or intercept-only when S is empty)
    * **Full model**: Y ~ Nyström(X, S)

    An F-test is performed on each cross-validation fold and the resulting
    p-values are averaged.

    Parameters
    ----------
    data, x, y, conditioning_set, alpha:
        As in :func:`fisher_z_test`.
    max_num_components:
        Maximum number of Nyström kernel approximation components.
    k_folds:
        Number of cross-validation folds.

    Returns
    -------
    CITestResult
    """
    cond = list(conditioning_set)
    df = data[[x, y] + cond].dropna()
    n = len(df)

    if n < k_folds + 2:
        return CITestResult(
            x=x, y=y, conditioning_set=frozenset(cond),
            p_value=float("nan"), statistic=float("nan"),
            is_independent=True,
            method=IndependenceTestMethod.REGRESSION_BASED,
        )

    y_arr = df[y].to_numpy(dtype=float)
    x_arr = df[[x]].to_numpy(dtype=float)
    z_arr = df[cond].to_numpy(dtype=float) if cond else None

    # Scale input features to improve Nyström numerical stability.
    x_arr = scale(x_arr)
    if z_arr is not None:
        z_arr = scale(z_arr)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    p_values: list[float] = []

    for train_idx, test_idx in kf.split(df):
        n_train = len(train_idx)
        y_train = y_arr[train_idx]

        n_comp = min(max_num_components, n_train // 3, n_train - 2)
        if n_comp < 1:
            n_comp = 1

        # Build full features (X + Z)
        if z_arr is not None:
            xz_train = np.column_stack([x_arr[train_idx], z_arr[train_idx]])
        else:
            xz_train = x_arr[train_idx]

        nc_full = min(n_comp, n_train - 2)
        if nc_full < 1:
            nc_full = 1

        nys_full = Nystroem(kernel="rbf", n_components=nc_full, random_state=42)
        phi_full = nys_full.fit_transform(xz_train)

        # Add intercept
        phi_full = np.column_stack([np.ones(n_train), phi_full])

        # Full model fit (in-sample)
        coef_full, _, _, _ = np.linalg.lstsq(phi_full, y_train, rcond=None)
        y_pred_full = phi_full @ coef_full
        rss_full = float(np.sum((y_train - y_pred_full) ** 2))
        df_full = phi_full.shape[1]

        # Reduced model fit
        if z_arr is not None:
            nc_red = min(max(n_comp // 2, 1), n_train - 2)
            nys_red = Nystroem(kernel="rbf", n_components=nc_red, random_state=42)
            phi_red = nys_red.fit_transform(z_arr[train_idx])
            phi_red = np.column_stack([np.ones(n_train), phi_red])
        else:
            # Intercept-only model
            phi_red = np.ones((n_train, 1))

        coef_red, _, _, _ = np.linalg.lstsq(phi_red, y_train, rcond=None)
        y_pred_red = phi_red @ coef_red
        rss_red = float(np.sum((y_train - y_pred_red) ** 2))
        df_red = phi_red.shape[1]

        p_val = _f_test_p_value(rss_red, rss_full, df_red, df_full, n_train)
        p_values.append(p_val)

    avg_p = float(np.mean(p_values))

    return CITestResult(
        x=x, y=y, conditioning_set=frozenset(cond),
        p_value=avg_p, statistic=float("nan"),
        is_independent=avg_p > alpha,
        method=IndependenceTestMethod.REGRESSION_BASED,
    )


# ---------------------------------------------------------------------------
# Generalised Covariance Measure (GCM) test
# ---------------------------------------------------------------------------


def generalised_cov_based_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    conditioning_set: Collection[str],
    alpha: float = 0.05,
) -> CITestResult:
    """Generalised Covariance Measure (GCM) conditional independence test.

    When a conditioning set is given, non-linear residuals are computed by
    regressing X on Z and Y on Z using Nyström-augmented OLS.  Independence is
    then tested via the normalised product of residuals: under the null, the
    statistic follows a standard normal distribution.

    When the conditioning set is empty, the ordinary Pearson correlation
    significance test is used as a lightweight fallback.

    Parameters
    ----------
    data, x, y, conditioning_set, alpha:
        As in :func:`fisher_z_test`.

    Returns
    -------
    CITestResult
    """
    cond = list(conditioning_set)
    df = data[[x, y] + cond].dropna()
    n = len(df)

    if n < 5:
        return CITestResult(
            x=x, y=y, conditioning_set=frozenset(cond),
            p_value=float("nan"), statistic=float("nan"),
            is_independent=True,
            method=IndependenceTestMethod.GENERALISED_COV_BASED,
        )

    x_arr = df[x].to_numpy(dtype=float)
    y_arr = df[y].to_numpy(dtype=float)

    if not cond:
        # Unconditional: residuals = centred values
        res_x = x_arr - x_arr.mean()
        res_y = y_arr - y_arr.mean()
    else:
        z_arr = df[cond].to_numpy(dtype=float)
        n_comp = min(40, n // 3, n - 1)
        if n_comp < 1:
            n_comp = 1

        nys = Nystroem(kernel="rbf", n_components=n_comp, random_state=42)
        z_feat = nys.fit_transform(z_arr)
        z_aug = np.column_stack([np.ones(n), z_feat])

        coef_x, _, _, _ = np.linalg.lstsq(z_aug, x_arr, rcond=None)
        res_x = x_arr - z_aug @ coef_x

        coef_y, _, _, _ = np.linalg.lstsq(z_aug, y_arr, rcond=None)
        res_y = y_arr - z_aug @ coef_y

    # GCM test statistic
    r_prod = res_x * res_y
    std_r = float(np.std(r_prod, ddof=1))

    if std_r < 1e-12:
        # Constant product ⇒ no evidence of dependence
        return CITestResult(
            x=x, y=y, conditioning_set=frozenset(cond),
            p_value=1.0, statistic=0.0,
            is_independent=True,
            method=IndependenceTestMethod.GENERALISED_COV_BASED,
        )

    test_stat = float(np.sum(r_prod) / np.sqrt(n)) / std_r
    p_value = float(2.0 * stats.norm.sf(abs(test_stat)))

    return CITestResult(
        x=x, y=y, conditioning_set=frozenset(cond),
        p_value=p_value, statistic=test_stat,
        is_independent=p_value > alpha,
        method=IndependenceTestMethod.GENERALISED_COV_BASED,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def run_ci_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    conditioning_set: Collection[str],
    method: IndependenceTestMethod = IndependenceTestMethod.FISHER_Z,
    alpha: float = 0.05,
    **kwargs,
) -> CITestResult:
    """Run a conditional independence test and return a :class:`CITestResult`.

    Parameters
    ----------
    data:
        Observed data.
    x, y:
        Names of the two variables to test.
    conditioning_set:
        Names of the conditioning variables.
    method:
        Which CI test to use.
    alpha:
        Significance level passed down to the chosen test.
    **kwargs:
        Extra keyword arguments forwarded to the underlying test function.

    Returns
    -------
    CITestResult
    """
    if method is IndependenceTestMethod.FISHER_Z:
        return fisher_z_test(data, x, y, conditioning_set, alpha=alpha, **kwargs)
    if method is IndependenceTestMethod.MUTUAL_INFO:
        return mutual_info_test(data, x, y, conditioning_set, alpha=alpha, **kwargs)
    if method is IndependenceTestMethod.KERNEL_BASED:
        return kernel_based_test(data, x, y, conditioning_set, alpha=alpha, **kwargs)
    if method is IndependenceTestMethod.APPROX_KERNEL_BASED:
        return approx_kernel_based_test(data, x, y, conditioning_set, alpha=alpha, **kwargs)
    if method is IndependenceTestMethod.REGRESSION_BASED:
        return regression_based_test(data, x, y, conditioning_set, alpha=alpha, **kwargs)
    if method is IndependenceTestMethod.GENERALISED_COV_BASED:
        return generalised_cov_based_test(data, x, y, conditioning_set, alpha=alpha, **kwargs)
    raise ValueError(f"Unknown CI test method: {method!r}")
