# Copyright (c) 2019, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import numpy as np
from quaternion.numba_wrapper import jit


@jit
def transition_function(x, x0, x1, y0=0.0, y1=1.0):
    """Return a smooth function that is constant outside (x0, x1).

    This uses the standard smooth (C^infinity) function with derivatives of compact support to
    transition between the two values, being constant outside of the transition region (x0, x1).

    Parameters
    ==========
    x: array_like
        One-dimensional monotonic array of floats.
    x0: float
        Value before which the output will equal `y0`.
    x1: float
        Value after which the output will equal `y1`.
    y0: float [defaults to 0.0]
        Value of the output before `x0`.
    y1: float [defaults to 1.0]
        Value of the output after `x1`.

    """
    transition = np.empty_like(x)
    ydiff = y1-y0
    i = 0
    while x[i] <= x0:
        i += 1
    transition[:i] = y0
    while x[i] < x1:
        tau = (x[i] - x0) / (x1 - x0)
        transition[i] = y0 + ydiff / (1.0 + np.exp(1.0/tau - 1.0/(1.0-tau)))
        i += 1
    transition[i:] = y1
    return transition


@jit
def bump_function(x, x0, x1, x2, x3, y0=0.0, y12=1.0, y3=0.0):
    """Return a smooth bump function that is constant outside (x0, x3) and inside (x1, x2).

    This uses the standard C^infinity function with derivatives of compact support to transition
    between the the given values.  By default, this is a standard bump function that is 0 outside of
    (x0, x3), and is 1 inside (x1, x2), but the constant values can all be adjusted optionally.

    Parameters
    ==========
    x: array_like
        One-dimensional monotonic array of floats.
    x0: float
        Value before which the output will equal `y0`.
    x1, x2: float
        Values between which the output will equal `y12`.
    x3: float
        Value after which the output will equal `y3`.
    y0: float [defaults to 0.0]
        Value of the output before `x0`.
    y12: float [defaults to 1.0]
        Value of the output after `x1` but before `x2`.
    y3: float [defaults to 0.0]
        Value of the output after `x3`.

    """
    bump = np.empty_like(x)
    ydiff01 = y12-y0
    ydiff23 = y3-y12
    i = 0
    while x[i] <= x0:
        i += 1
    bump[:i] = y0
    while x[i] < x1:
        tau = (x[i] - x0) / (x1 - x0)
        bump[i] = y0 + ydiff01 / (1.0 + np.exp(1.0/tau - 1.0/(1.0-tau)))
        i += 1
    i1 = i
    while x[i] <= x2:
        i += 1
    bump[i1:i] = y12
    while x[i] < x3:
        tau = (x[i] - x2) / (x3 - x2)
        bump[i] = y12 + ydiff23 / (1.0 + np.exp(1.0/tau - 1.0/(1.0-tau)))
        i += 1
    bump[i:] = y3
    return bump
