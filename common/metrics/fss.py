# Implementation of FSS (Fractions Skill Score) from pysteps.verification.spatialscores.fss
# https://github.com/pySTEPS/pysteps/blob/master/pysteps/verification/spatialscores.py

import numpy as np
from scipy.ndimage import uniform_filter

def fss(X_f, X_o, thr, scale):
    """
    Compute the fractions skill score (FSS) for a deterministic forecast field
    and the corresponding observation field.

    Parameters
    ----------
    X_f: array_like
        Array of shape (m, n) containing the forecast field.
    X_o: array_like
        Array of shape (m, n) containing the observation field.
    thr: float
        The intensity threshold.
    scale: int
        The spatial scale in pixels. In practice, the scale represents the size
        of the moving window that it is used to compute the fraction of pixels
        above the threshold.

    Returns
    -------
    out: float
        The fractions skill score between 0 and 1.

    References
    ----------
    :cite:`RL2008`, :cite:`EWWM2013`
    """

    fss = fss_init(thr, scale)
    fss_accum(fss, X_f, X_o)
    return fss_compute(fss)


def fss_init(thr, scale):
    """
    Initialize a fractions skill score (FSS) verification object.

    Parameters
    ----------
    thr: float
        The intensity threshold.
    scale: float
        The spatial scale in pixels. In practice, the scale represents the size
        of the moving window that it is used to compute the fraction of pixels
        above the threshold.

    Returns
    -------
    fss: dict
        The initialized FSS verification object.
    """
    fss = dict(thr=thr, scale=scale, sum_fct_sq=0.0, sum_fct_obs=0.0, sum_obs_sq=0.0)

    return fss


def fss_accum(fss, X_f, X_o):
    """Accumulate forecast-observation pairs to an FSS object.

    Parameters
    -----------
    fss: dict
        The FSS object initialized with
        :py:func:`pysteps.verification.spatialscores.fss_init`.
    X_f: array_like
        Array of shape (m, n) containing the forecast field.
    X_o: array_like
        Array of shape (m, n) containing the observation field.
    """
    if len(X_f.shape) != 2 or len(X_o.shape) != 2 or X_f.shape != X_o.shape:
        message = "X_f and X_o must be two-dimensional arrays"
        message += " having the same shape"
        raise ValueError(message)

    X_f = X_f.copy()
    X_f[~np.isfinite(X_f)] = fss["thr"] - 1
    X_o = X_o.copy()
    X_o[~np.isfinite(X_o)] = fss["thr"] - 1

    # Convert to binary fields with the given intensity threshold
    I_f = (X_f >= fss["thr"]).astype(float)
    I_o = (X_o >= fss["thr"]).astype(float)

    # Compute fractions of pixels above the threshold within a square
    # neighboring area by applying a 2D moving average to the binary fields
    if fss["scale"] > 1:
        S_f = uniform_filter(I_f, size=fss["scale"], mode="constant", cval=0.0)
        S_o = uniform_filter(I_o, size=fss["scale"], mode="constant", cval=0.0)
    else:
        S_f = I_f
        S_o = I_o

    fss["sum_obs_sq"] += np.nansum(S_o**2)
    fss["sum_fct_obs"] += np.nansum(S_f * S_o)
    fss["sum_fct_sq"] += np.nansum(S_f**2)


def fss_merge(fss_1, fss_2):
    """
    Merge two FSS objects.

    Parameters
    ----------
    fss_1: dict
      A FSS object initialized with
      :py:func:`pysteps.verification.spatialscores.fss_init`.
      and populated with
      :py:func:`pysteps.verification.spatialscores.fss_accum`.
    fss_2: dict
      Another FSS object initialized with
      :py:func:`pysteps.verification.spatialscores.fss_init`.
      and populated with
      :py:func:`pysteps.verification.spatialscores.fss_accum`.

    Returns
    -------
    out: dict
      The merged FSS object.
    """

    # checks
    if fss_1["thr"] != fss_2["thr"]:
        raise ValueError(
            "cannot merge: the thresholds are not same %s!=%s"
            % (fss_1["thr"], fss_2["thr"])
        )
    if fss_1["scale"] != fss_2["scale"]:
        raise ValueError(
            "cannot merge: the scales are not same %s!=%s"
            % (fss_1["scale"], fss_2["scale"])
        )

    # merge the FSS objects
    fss = fss_1.copy()
    fss["sum_obs_sq"] += fss_2["sum_obs_sq"]
    fss["sum_fct_obs"] += fss_2["sum_fct_obs"]
    fss["sum_fct_sq"] += fss_2["sum_fct_sq"]

    return fss


def fss_compute(fss):
    """
    Compute the FSS.

    Parameters
    ----------
    fss: dict
       An FSS object initialized with
       :py:func:`pysteps.verification.spatialscores.fss_init`
       and accumulated with
       :py:func:`pysteps.verification.spatialscores.fss_accum`.

    Returns
    -------
    out: float
        The computed FSS value.
    """
    numer = fss["sum_fct_sq"] - 2.0 * fss["sum_fct_obs"] + fss["sum_obs_sq"]
    denom = fss["sum_fct_sq"] + fss["sum_obs_sq"]

    return 1.0 - numer / denom