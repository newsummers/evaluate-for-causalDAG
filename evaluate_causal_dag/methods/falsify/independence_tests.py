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
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Collection, FrozenSet, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


class IndependenceTestMethod(str, Enum):
    """Enumeration of available CI test methods."""

    FISHER_Z = "fisher_z"
    MUTUAL_INFO = "mutual_info"


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
    raise ValueError(f"Unknown CI test method: {method!r}")
