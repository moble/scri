# Copyright (c) 2019, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import functools
import numpy as np
from quaternion.numba_wrapper import njit


def swsh_indices_to_matrix_indices(matrix_iterator):
    """Convert SWSH-indexed function into caching sparse matrix-indexed function.

    This function is designed to decorate functions with signature (ell_min, ell_max) that yield
    tuples of (ellp, mp, ell, m, value), and convert them into functions with the same signature but
    returning tuples of (rows, columns, values).  Here, the rows are row indices of a matrix,
    corresponding to the (ellp, mp) in standard `spherical_functions` order.  Similarly the columns
    are column indices corresponding to (ell, m).  The result can be passed into the
    `sparse_expectation_value` function.

    """
    from spherical_functions import LM_index

    @functools.lru_cache()  # Decorate the decorated function with caching
    def wrapper(ell_min, ell_max, *args, **kwargs):
        rows, columns, values = zip(*(  # zip* is the inverse of zip
            (LM_index(ellp, mp, ell_min), LM_index(ell, m, ell_min), value)
            for ellp, mp, ell, m, value in matrix_iterator(ell_min, ell_max, *args, **kwargs)
        ))
        return np.array(rows, dtype=int), np.array(columns, dtype=int), np.array(values)

    functools.update_wrapper(wrapper, matrix_iterator)  # Copy over the name, docstring, and such

    return wrapper


@njit
def sparse_expectation_value(abar, rows, columns, values, b):
    """Low-level numba-friendly helper function for the main
    calculation of `matrix_expectation_value`.

    Computes <a|M|b>, assuming that <a| and |b> have the same shapes, and that (rows, columns,
    values) are corresponding arrays containing the nonzero elements of the matrix M.

    Parameters
    ----------
    abar : ndarray
        The data for <a| (after complex conjugation was applied).

    rows, columns : list
        Lists of same length, containing indices into (respectively)
        abar and b.

    values : ndarray
        Should have shape (n,) where n is the same length as rows and
        columns.  These values are the matrix elements themselves to
        be summed.

    b : ndarray
        The data for |b>. Must have the same shape as abar.

    Returns
    -------
    expectation_value : ndarray

    """
    n_times = abar.shape[0]
    n_elements = rows.shape[0]
    expectation_value = np.zeros(n_times, dtype=np.complex_)
    for i_time in range(n_times):
        for i_element in range(n_elements):
            expectation_value[i_time] += (
                abar[i_time, rows[i_element]]
                * b[i_time, columns[i_element]]
                * values[i_element]
            )
    return expectation_value


def matrix_expectation_value(a, M, b,
                             allow_LM_differ=False, allow_times_differ=False):
    """The matrix expectation value <a|M|b>(u), where M is a linear operator.

    Treat two spin-s waveforms a, b (in the modal representation) as vectors.
    Then we can form the 'expectation value'

    .. math:: <a|M|b>(u) \equiv \sum_{\ell', m', \ell, m} a^*_{\ell' m'}(u) M_{\ell' m' \ell m} b_{\ell m}(u) \,.

    For efficiency with sparse matrices, M should be implemented as a
    generator which yields tuples (ellp, mp, ell, m, M_{ellp mp ell m})
    for only the non-vanishing matrix elements.

    Parameters
    ----------
    a : WaveformModes object
        The waveform |a>. This function is "antilinear" in `a`.

    M : callable
        M will be called like
        `for ellp, mp, ell, m, M_el in M(ell_min, ell_max):`
        This is best implemented as a generator yielding tuples
        `(ellp, mp, ell, m, M_{ellp mp ell m})`
        which range over ell, ellp values up to and including ell_max,
        for only the non-vanishing matrix elements of M.

    b : WaveformModes object
        The waveform |b>. This function is linear in `b`.

    allow_LM_differ : bool, optional [default: False]
        If True and if the set of (ell,m) modes between a and b
        differ, then the inner product will be computed using the
        intersection of the set of modes.

    allow_times_differ: bool, optional [default: False]
        If True and if the set of times between a and b differ,
        then both WaveformModes will be interpolated to the
        intersection of the set of times.

    Returns
    -------
    times, expect_val : ndarray, complex ndarray
        Resulting tuple has length two.  The first element is the set
        of times u for the timeseries.  The second element is the
        timeseries of <a|M|b>(u).

    """
    import numpy as np
    from .extrapolation import intersection
    from spherical_functions import LM_index

    if (a.spin_weight != b.spin_weight):
        raise ValueError("Spin weights must match in matrix_expectation_value")

    LM_clip = slice(a.ell_min, a.ell_max + 1)
    if ((a.ell_min != b.ell_min) or (a.ell_max != b.ell_max)):
        if (allow_LM_differ):
            LM_clip = slice( max(a.ell_min, b.ell_min),
                             min(a.ell_max, b.ell_max) + 1 )
            if (LM_clip.start >= LM_clip.stop):
                raise ValueError("Intersection of (ell,m) modes is "
                                 "empty.  Assuming this is not desired.")
        else:
            raise ValueError("ell_min and ell_max must match in matrix_expectation_value "
                             "(use allow_LM_differ=True to override)")

    t_clip = None
    if not np.array_equal(a.t, b.t):
        if (allow_times_differ):
            t_clip = intersection(a.t, b.t)
        else:
            raise ValueError("Time samples must match in matrix_expectation_value "
                             "(use allow_times_differ=True to override)")

    ##########

    times = a.t
    A = a
    B = b

    if (LM_clip is not None):
        A = A[:,LM_clip]
        B = B[:,LM_clip]

    if (t_clip is not None):
        times = t_clip
        A = A.interpolate(t_clip)
        B = B.interpolate(t_clip)

    ##########
    ## Time for actual math!

    Abar_data  = np.conj(A.data)
    B_data     = B.data

    ell_min = LM_clip.start
    ell_max = LM_clip.stop - 1

    rows, columns, values = M(ell_min, ell_max)

    return (times,
            sparse_expectation_value(Abar_data, rows, columns, values, B_data))


def energy_flux(h):
    """Compute energy flux from waveform

    This implements Eq. (2.8) from Ruiz+ (2008) [0707.4654].
    """
    import numpy as np
    from .waveform_modes import WaveformModes
    from . import h as htype
    from . import hdot as hdottype
    if not isinstance(h, WaveformModes):
        raise ValueError("Momentum flux can only be calculated from a `WaveformModes` object; "
                         +"this object is of type `{0}`.".format(type(h)))
    if h.dataType == hdottype:
        hdot = h.data
    elif h.dataType == htype:
        hdot = h.data_dot
    else:
        raise ValueError("Input argument is expected to have data of type `h` or `hdot`; "
                         +"this waveform data has type `{0}`".format(h.data_type_string))

    # No need to use matrix_expectation_value here
    Edot = np.einsum('ij, ij -> i', hdot.conjugate(), hdot).real

    Edot /= 16.*np.pi

    return Edot


@swsh_indices_to_matrix_indices
def p_z(ell_min, ell_max, s=-2):
    """Generator for p^z matrix elements (for use with matrix_expectation_value)

    This function is specific to the case where waveforms have s=-2.

    p^z = \cos\theta = 2 \sqrt{\pi/3} Y_{1,0}
    This is what Ruiz+ (2008) [0707.4654] calls "l^z", which is a bad name.

    The matrix elements yielded are
    < s, ellp, mp | \cos\theta | s, ell, m > =
      \sqrt{ \frac{2*ell+1}{2*ellp+1} } *
      < ell, m, 1, 0 | ellp, m > < ell, -s, 1, 0 | ellp, -s >
    where the terms on the last line are the ordinary Clebsch-Gordan coefficients.
    Because of the magnetic selection rules, we only have mp == m

    We could have used `_swsh_Y_mat_el` but I am just preemptively
    combining the prefactors.
    """
    import numpy as np
    from spherical_functions import clebsch_gordan as CG

    for ell in range(ell_min, ell_max+1):
        ellp_min = max(ell_min, ell - 1)
        ellp_max = min(ell_max, ell + 1)
        for ellp in range(ellp_min, ellp_max+1):
            for m in range(-ell, ell+1):
                if ((m < -ellp) or (m > ellp)):
                    continue
                cg1 = CG(ell, m,  1, 0, ellp, m)
                cg2 = CG(ell, -s, 1, 0, ellp, -s)
                prefac = np.sqrt( (2.*ell + 1.) / (2.*ellp + 1.) )
                yield ellp, m, ell, m, (prefac * cg1 * cg2)


@swsh_indices_to_matrix_indices
def p_plusminus(ell_min, ell_max, sign, s=-2):
    u"""Produce the function p_plus or p_minus, based on sign.

      p^+ = -\sqrt{8 \pi / 3} Y_{1,+1} = \sin\theta e^{+i\phi}
      p^- = +\sqrt{8 \pi / 3} Y_{1,-1} = \sin\theta e^{-i\phi}

    This is what Ruiz+ (2008) [0707.4654] calls "l^±", which is a confusing name.

    We use `swsh_Y_mat_el` to compute the matrix elements.  Notice that since the operator has
    definite m = ±1, we only have mp == m ± 1 nonvanishing in the matrix elements.

    """
    import numpy as np

    if (sign != 1) and (sign != -1):
        raise ValueError("sign must be either 1 or -1 in j_plusminus")

    prefac = -1. * sign * np.sqrt( 8. * np.pi / 3. )

    def swsh_Y_mat_el(s, l3, m3, l1, m1, l2, m2):
        """Compute a matrix element treating Y_{\ell, m} as a linear operator

        From the rules for the Wigner D matrices, we get the result that
        <s, l3, m3 | Y_{l1, m1} | s, l2, m2 > =
          \sqrt{ \frac{(2*l1+1)(2*l2+1)}{4*\pi*(2*l3+1)} } *
          < l1, m1, l2, m2 | l3, m3 > < l1, 0, l2, −s | l3, −s >
        where the terms on the last line are the ordinary Clebsch-Gordan coefficients.
        See e.g. Campbell and Morgan (1971).
        """
        from spherical_functions import clebsch_gordan as CG

        cg1 = CG(l1, m1, l2, m2, l3, m3)
        cg2 = CG(l1, 0., l2, -s, l3, -s)

        return np.sqrt( (2.*l1 + 1.) * (2.*l2 + 1.) / (4. * np.pi * (2.*l3 + 1)) ) * cg1 * cg2

    for ell in range(ell_min, ell_max+1):
        ellp_min = max(ell_min, ell - 1)
        ellp_max = min(ell_max, ell + 1)
        for ellp in range(ellp_min, ellp_max+1):
            for m in range(-ell, ell+1):
                mp = round(m + 1 * sign)
                if ((mp < -ellp) or (mp > ellp)):
                    continue
                yield (ellp, mp, ell, m,
                       (prefac *
                        swsh_Y_mat_el(s, ellp, mp, 1., sign, ell, m)))


p_plus  = functools.partial(p_plusminus, sign=+1)
p_minus = functools.partial(p_plusminus, sign=-1)
p_plus.__doc__  = p_plusminus.__doc__
p_minus.__doc__ = p_plusminus.__doc__


def momentum_flux(h):
    """Compute momentum flux from waveform

    This implements Eq. (2.11) from Ruiz+ (2008) [0707.4654] by using `matrix_expectation_value`
    with `p_z`, `p_plus`, and `p_minus`.

    """
    import numpy as np
    from .waveform_modes import WaveformModes
    from . import h as htype
    from . import hdot as hdottype

    if not isinstance(h, WaveformModes):
        raise ValueError("Momentum flux can only be calculated from a `WaveformModes` object; "
                         +"this object is of type `{0}`.".format(type(h)))
    if h.dataType == hdottype:
        hdot = h
    elif h.dataType == htype:
        hdot = h.copy()
        hdot.dataType = hdottype
        hdot.data = h.data_dot
    else:
        raise ValueError("Input argument is expected to have data of type `h` or `hdot`; "
                         +"this waveform data has type `{0}`".format(h.data_type_string))

    pdot = np.zeros((hdot.n_times, 3), dtype=float)

    _, p_plus_dot  = matrix_expectation_value( hdot, functools.partial(p_plus, s=-2),  hdot )
    _, p_minus_dot = matrix_expectation_value( hdot, functools.partial(p_minus, s=-2), hdot )
    _, p_z_dot = matrix_expectation_value( hdot, functools.partial(p_z, s=-2), hdot )

    # Convert into (x,y,z) basis
    pdot[:,0] = 0.5 * ( p_plus_dot.real + p_minus_dot.real )
    pdot[:,1] = 0.5 * ( p_plus_dot.imag - p_minus_dot.imag )
    pdot[:,2] = p_z_dot.real

    pdot /= 16.*np.pi

    return pdot


@swsh_indices_to_matrix_indices
def j_z(ell_min, ell_max):
    """Generator for j^z matrix elements (for use with matrix_expectation_value)

    Matrix elements yielded are

      <ellp, mp|j^z|ell, m> = 1.j * m \delta_{ellp ell} \delta_{mp, m}

    This follows the convention for j^z from Ruiz+ (2008) [0707.4654]

    """
    for ell in range(ell_min, ell_max+1):
        for m in range(-ell, ell+1):
            yield ell, m, ell, m, (1.j * m)


@swsh_indices_to_matrix_indices
def j_plusminus(ell_min, ell_max, sign):
    """Produce the function j_plus or j_minus, based on sign.

    The conventions for these matrix elements, to agree with Ruiz+ (2008) [0707.4654], should be:

      <ellp, mp|j^+|ell, m> = 1.j * np.sqrt((l-m)(l+m+1)) \delta{ellp, ell} \delta{mp, (m+1)}
      <ellp, mp|j^-|ell, m> = 1.j * np.sqrt((l+m)(l-m+1)) \delta{ellp, ell} \delta{mp, (m-1)}

    The spinsfast function ladder_operator_coefficient(l, m) gives np.sqrt((l-m)(l+m+1)).

    """
    from spherical_functions import ladder_operator_coefficient as ladder

    if (sign != 1) and (sign != -1):
        raise ValueError("sign must be either 1 or -1 in j_plusminus")

    for ell in range(ell_min, ell_max+1):
        for m in range(-ell, ell+1):
            mp = round(m + 1 * sign)
            if ((mp < -ell) or (mp > ell)):
                continue
            yield ell, mp, ell, m, (1.j * ladder(ell, m*sign))


j_plus  = functools.partial(j_plusminus, sign=+1)
j_minus = functools.partial(j_plusminus, sign=-1)
j_plus.__doc__  = j_plusminus.__doc__
j_minus.__doc__ = j_plusminus.__doc__


def angular_momentum_flux(h, hdot=None):
    """Compute angular momentum flux from waveform

    This implements Eq. (2.24) from Ruiz+ (2008) [0707.4654] by using `matrix_expectation_value`
    with (internal) helper functions `j_z`, `j_plus`, and `j_minus`.

    """
    import numpy as np
    from .waveform_modes import WaveformModes
    from . import h as htype
    from . import hdot as hdottype
    if not isinstance(h, WaveformModes):
        raise ValueError("Angular momentum flux can only be calculated from a `WaveformModes` object; "
                         +"`h` is of type `{0}`.".format(type(h)))
    if (hdot is not None) and (not isinstance(hdot, WaveformModes)):
        raise ValueError("Angular momentum flux can only be calculated from a `WaveformModes` object; "
                         +"`hdot` is of type `{0}`.".format(type(hdot)))
    if (h.dataType == htype):
        if (hdot is None):
            hdot = h.copy()
            hdot.dataType = hdottype
            hdot.data = h.data_dot
        elif (hdot.dataType != hdottype):
            raise ValueError("Input argument `hdot` is expected to have data of type `hdot`; "
                             +"this `hdot` waveform data has type `{0}`".format(h.data_type_string))
    else:
        raise ValueError("Input argument `h` is expected to have data of type `h`; "
                         +"this `h` waveform data has type `{0}`".format(h.data_type_string))

    jdot = np.zeros((hdot.n_times, 3), dtype=float)

    _, j_plus_dot  = matrix_expectation_value( hdot, j_plus,  h )
    _, j_minus_dot = matrix_expectation_value( hdot, j_minus, h )
    _, j_z_dot     = matrix_expectation_value( hdot, j_z,     h )

    # Convert into (x,y,z) basis
    jdot[:,0] = 0.5 * ( j_plus_dot.real + j_minus_dot.real )
    jdot[:,1] = 0.5 * ( j_plus_dot.imag - j_minus_dot.imag )
    jdot[:,2] = j_z_dot.real

    jdot /= -16.*np.pi

    return jdot


def poincare_fluxes(h, hdot=None):
    """Compute fluxes of energy, momentum, and angular momemntum. This
    function will compute the time derivative 1 or 0 times (if an
    optional argument is passed) so is more efficient than separate
    calls to `energy_flux`, `momentum_flux`, and
    `angular_momentum_flux`.

    Parameters
    ----------
    h : WaveformModes object
        Must have type `h`

    hdot : WaveformModes object, optional [default: None]
        The time derivative of h.  If None, computed from h.  Must
        have type `hdot`.

    Returns
    -------
    edot, pdot, jdot : ndarray, ndarray, ndarray

    """
    from .waveform_modes import WaveformModes
    from . import h as htype
    from . import hdot as hdottype
    if not isinstance(h, WaveformModes):
        raise ValueError("Poincare fluxes can only be calculated from a `WaveformModes` object; "
                         +"`h` is of type `{0}`.".format(type(h)))
    if (hdot is not None) and (not isinstance(hdot, WaveformModes)):
        raise ValueError("Poincare fluxes can only be calculated from a `WaveformModes` object; "
                         +"`hdot` is of type `{0}`.".format(type(hdot)))
    if (h.dataType == htype):
        if (hdot is None):
            hdot = h.copy()
            hdot.dataType = hdottype
            hdot.data = h.data_dot
        elif (hdot.dataType != hdottype):
            raise ValueError("Input argument `hdot` is expected to have data of type `hdot`; "
                             +"this `hdot` waveform data has type `{0}`".format(h.data_type_string))
    else:
        raise ValueError("Input argument `h` is expected to have data of type `h`; "
                         +"this `h` waveform data has type `{0}`".format(h.data_type_string))

    return (energy_flux(hdot),
            momentum_flux(hdot),
            angular_momentum_flux(h, hdot))
