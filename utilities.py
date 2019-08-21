# Copyright (c) 2019, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import numpy as np
from quaternion.numba_wrapper import jit

maxexp = np.finfo(float).maxexp * np.log(2) * 0.99


@jit
def _transition_function(x, x0, x1, y0, y1):
    transition = np.empty_like(x)
    ydiff = y1-y0
    i = 0
    while x[i] <= x0:
        i += 1
    i0 = i
    transition[:i0] = y0
    while x[i] < x1:
        tau = (x[i] - x0) / (x1 - x0)
        exponent = 1.0/tau - 1.0/(1.0-tau)
        if exponent >= maxexp:
            transition[i] = y0
        else:
            transition[i] = y0 + ydiff / (1.0 + np.exp(exponent))
        i += 1
    i1 = i
    transition[i1:] = y1
    return transition, i0, i1


def transition_function(x, x0, x1, y0=0.0, y1=1.0, return_indices=False):
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
    return_indices: bool [defaults to False]
        If True, return the array and the indices (i0, i1) at which the transition occurs, such that
        t[:i0]==y0 and t[i1:]==y1.

    """
    if return_indices:
        return _transition_function(x, x0, x1, y0, y1)
    return _transition_function(x, x0, x1, y0, y1)[0]


@jit
def transition_function_derivative(x, x0, x1, y0=0.0, y1=1.0):
    """Return derivative of the transition function

    This function simply returns the derivative of `transition_function` with respect to the `x`
    parameter.  The parameters to this function are identical to those of that function.

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
    transition_prime = np.zeros_like(x)
    ydiff = y1-y0
    i = 0
    while x[i] <= x0:
        i += 1
    while x[i] < x1:
        tau = (x[i] - x0) / (x1 - x0)
        exponent = 1.0/tau - 1.0/(1.0-tau)
        if exponent >= maxexp:
            transition_prime[i] = 0.0
        else:
            exponential = np.exp(1.0/tau - 1.0/(1.0-tau))
            transition_prime[i] = -ydiff * exponential * (-1.0/tau**2 - 1.0/(1.0-tau)**2) * (1/(x1 - x0)) / (1.0 + exponential)**2
        i += 1
    return transition_prime


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
        exponent = 1.0/tau - 1.0/(1.0-tau)
        if exponent >= maxexp:
            bump[i] = y0
        else:
            bump[i] = y0 + ydiff01 / (1.0 + np.exp(exponent))
        i += 1
    i1 = i
    while x[i] <= x2:
        i += 1
    bump[i1:i] = y12
    while x[i] < x3:
        tau = (x[i] - x2) / (x3 - x2)
        exponent = 1.0/tau - 1.0/(1.0-tau)
        if exponent >= maxexp:
            bump[i] = y12
        else:
            bump[i] = y12 + ydiff23 / (1.0 + np.exp(exponent))
        i += 1
    bump[i:] = y3
    return bump


def transition_to_constant(f, t, t1, t2):
    """Smoothly transition from the function to a constant.
    
    This works (implicitly) by multiplying the derivative of `f` with the transition function, and
    then integrating.  Using integration by parts, this simplifies to multiplying `f` itself by the
    transition function, and then subtracting the integral of `f` times the derivative of the
    transition function.  This integral is effectively restricted to the region (t1, t2).  Note that
    the final value (after t2) will depend on the precise values of `t1` and `t2`, and the behavior
    of `f` in between.

    Parameters
    ==========
    f: array_like
        One-dimensional array corresponding to the following `t` parameter.
    t: array_like
        One-dimensional monotonic array of floats.
    t1: float
        Value before which the output will equal `f`.
    t2: float
        Value after which the output will be constant.

    """
    from quaternion import indefinite_integral
    transition, i1, i2 = transition_function(t, t1, t2, y0=1.0, y1=0.0, return_indices=True)
    transition_dot = transition_function_derivative(t, t1, t2, y0=1.0, y1=0.0)
    f_transitioned = f * transition
    f_transitioned[i1:i2] -= indefinite_integral(f[i1:i2] * transition_dot[i1:i2], t[i1:i2])
    f_transitioned[i2:] = f_transitioned[i2-1]
    return f_transitioned
