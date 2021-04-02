# Copyright (c) 2019, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import functools
import numpy as np
import numba
from spherical_functions import clebsch_gordan as CG
from . import jit


def swsh_indices_to_matrix_indices(matrix_iterator):
    """Convert SWSH-indexed function into caching sparse matrix-indexed function

    This function is designed to decorate functions with signature (ell_min,
    ell_max) that yield tuples of (ellp, mp, ell, m, value), and convert them into
    functions with the same signature but returning tuples of (rows, columns,
    values).  Here, the rows are row indices of a matrix, corresponding to the
    (ellp, mp) in standard `spherical_functions` order.  Similarly the columns are
    column indices corresponding to (ell, m).  The result can be passed into the
    `sparse_expectation_value` function.

    """
    from spherical_functions import LM_index

    @functools.lru_cache()  # Decorate the decorated function with caching
    def wrapper(ell_min, ell_max, *args, **kwargs):
        rows, columns, values = zip(
            *(  # zip* is the inverse of zip
                (LM_index(ellp, mp, ell_min), LM_index(ell, m, ell_min), value)
                for ellp, mp, ell, m, value in matrix_iterator(ell_min, ell_max, *args, **kwargs)
            )
        )
        return np.array(rows, dtype=int), np.array(columns, dtype=int), np.array(values)

    functools.update_wrapper(wrapper, matrix_iterator)  # Copy over the name, docstring, and such

    return wrapper


@jit
def sparse_expectation_value(abar, rows, columns, values, b):
    """Helper function for the main calculation of `matrix_expectation_value`

    Computes ⟨a|M|b⟩, assuming that ⟨a| and |b⟩ have the same shapes, and that
    (rows, columns, values) are corresponding arrays containing the nonzero
    elements of the matrix M.

    Parameters
    ----------
    abar : ndarray
        The data for ⟨a| (after complex conjugation was applied).

    rows, columns : list
        Lists of same length, containing indices into (respectively)
        abar and b.

    values : ndarray
        Should have shape (n,) where n is the same length as rows and
        columns.  These values are the matrix elements themselves to
        be summed.

    b : ndarray
        The data for |b⟩. Must have the same shape as abar.

    Returns
    -------
    expectation_value : ndarray

    """
    n_times = abar.shape[0]
    n_elements = rows.shape[0]
    expectation_value = np.zeros(n_times, dtype=numba.complex128)
    for i_time in range(n_times):
        for i_element in range(n_elements):
            expectation_value[i_time] += (
                abar[i_time, rows[i_element]] * b[i_time, columns[i_element]] * values[i_element]
            )
    return expectation_value


def matrix_expectation_value(a, M, b, allow_LM_differ=False, allow_times_differ=False):
    """The matrix expectation value ⟨a|M|b⟩(u), where M is a linear operator

    Treat two spin-s waveforms a, b (in the modal representation) as vectors.
    Then we can form the 'expectation value'

      ⟨a|M|b⟩(u) ≡ ∑ⱼₙₗₘ āⱼₙ(u) Mⱼₙₗₘ bₗₘ(u).

    For efficiency with sparse matrices, M should be implemented as a generator
    which yields tuples (j, n, l, m, Mⱼₙₗₘ) for only the non-vanishing matrix
    elements.

    Parameters
    ----------
    a : WaveformModes object
        The waveform |a⟩. This function is "antilinear" in `a`.

    M : callable
        M will be called like
        `for ellp, mp, ell, m, M_el in M(ell_min, ell_max):`
        This is best implemented as a generator yielding tuples
        `(ellp, mp, ell, m, M_{ellp mp ell m})`
        which range over ell, ellp values up to and including ell_max,
        for only the non-vanishing matrix elements of M.

    b : WaveformModes object
        The waveform |b⟩. This function is linear in `b`.

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
        timeseries of ⟨a|M|b⟩(u).

    """
    from .extrapolation import intersection
    from spherical_functions import LM_index

    if a.spin_weight != b.spin_weight:
        raise ValueError("Spin weights must match in matrix_expectation_value")

    LM_clip = slice(a.ell_min, a.ell_max + 1)
    if (a.ell_min != b.ell_min) or (a.ell_max != b.ell_max):
        if allow_LM_differ:
            LM_clip = slice(max(a.ell_min, b.ell_min), min(a.ell_max, b.ell_max) + 1)
            if LM_clip.start >= LM_clip.stop:
                raise ValueError("Intersection of (ell,m) modes is " "empty.  Assuming this is not desired.")
        else:
            raise ValueError(
                "ell_min and ell_max must match in matrix_expectation_value " "(use allow_LM_differ=True to override)"
            )

    t_clip = None
    if not np.array_equal(a.t, b.t):
        if allow_times_differ:
            t_clip = intersection(a.t, b.t)
        else:
            raise ValueError(
                "Time samples must match in matrix_expectation_value " "(use allow_times_differ=True to override)"
            )

    ##########

    times = a.t
    A = a
    B = b

    if LM_clip is not None:
        A = A[:, LM_clip]
        B = B[:, LM_clip]

    if t_clip is not None:
        times = t_clip
        A = A.interpolate(t_clip)
        B = B.interpolate(t_clip)

    ##########
    ## Time for actual math!

    Abar_data = np.conj(A.data)
    B_data = B.data

    ell_min = LM_clip.start
    ell_max = LM_clip.stop - 1

    rows, columns, values = M(ell_min, ell_max)

    return (times, sparse_expectation_value(Abar_data, rows, columns, values, B_data))


def energy_flux(h):
    """Compute energy flux from waveform

    This implements Eq. (2.8) from Ruiz et al. (2008) [0707.4654].

    """
    from .waveform_modes import WaveformModes
    from . import h as htype
    from . import hdot as hdottype

    if not isinstance(h, WaveformModes):
        raise ValueError(
            f"Energy flux can only be calculated from a `WaveformModes` object; this object is of type `{type(h)}`."
        )
    if h.dataType == hdottype:
        hdot = h.data
    elif h.dataType == htype:
        hdot = h.data_dot
    else:
        raise ValueError(
            f"Input argument is expected to have data of type `h` or `hdot`; this waveform data has type `{h.data_type_string}`"
        )

    # No need to use matrix_expectation_value here
    Edot = np.einsum("ij, ij -> i", hdot.conjugate(), hdot).real

    Edot /= 16.0 * np.pi

    return Edot


@swsh_indices_to_matrix_indices
def p_z(ell_min, ell_max, s=-2):
    """Generator for pᶻ matrix elements (for use with `matrix_expectation_value`)

    This function is specific to the case where waveforms have s=-2.

      pᶻ = cosθ = 2 √(π/3) Y₁₀

    This is what Ruiz et al. (2008) [0707.4654] call "lᶻ".

    The matrix elements yielded are

      ⟨s,j,n|cosθ|s,l,m⟩ = √[(2l+1)/(2j+1)] ⟨l,m,1,0|j,m⟩ ⟨l,-s,1,0|j,-s⟩

    where the terms on the last line are the ordinary Clebsch-Gordan coefficients.
    Because of the magnetic selection rules, we only have nonzero elements for
    n==m.

    We could have used `_swsh_Y_mat_el` but I am just preemptively combining the
    prefactors.

    """

    for ell in range(ell_min, ell_max + 1):
        ellp_min = max(ell_min, ell - 1)
        ellp_max = min(ell_max, ell + 1)
        for ellp in range(ellp_min, ellp_max + 1):
            for m in range(-ell, ell + 1):
                if (m < -ellp) or (m > ellp):
                    continue
                cg1 = CG(ell, m, 1, 0, ellp, m)
                cg2 = CG(ell, -s, 1, 0, ellp, -s)
                prefac = np.sqrt((2.0 * ell + 1.0) / (2.0 * ellp + 1.0))
                yield ellp, m, ell, m, (prefac * cg1 * cg2)


@swsh_indices_to_matrix_indices
def p_plusminus(ell_min, ell_max, sign, s=-2):
    """Produce the function p_plus or p_minus, based on sign.

      p⁺ = -√(8π/3) Y₁₊₁ = sinθ exp[+iϕ]
      p⁻ = +√(8π/3) Y₁₋₁ = sinθ exp[-iϕ]

    These are what Ruiz et al. (2008) [0707.4654] call "l⁺" and "l⁻".

    We use `swsh_Y_mat_el` to compute the matrix elements.  Notice that since the
    operator has definite m=±1, we only have mp==m±1 nonvanishing in the
    matrix elements.

    """

    if (sign != 1) and (sign != -1):
        raise ValueError("sign must be either 1 or -1 in j_plusminus")

    prefac = -1.0 * sign * np.sqrt(8.0 * np.pi / 3.0)

    def swsh_Y_mat_el(s, l3, m3, l1, m1, l2, m2):
        """Compute a matrix element treating Yₗₘ as a linear operator

        From the rules for the Wigner D matrices, we get the result that

          ⟨s,l₃,m₃|Yₗ₁ₘ₁|s,l₂,m₂⟩ =
            √[(2l₁+1)(2l₂+1)/(4π(2l₃+1))] ⟨l₁,m₁,l₂,m₂|l₃,m₃⟩ ⟨l₁,0,l₂,−s|l₃,−s⟩

        where the terms on the last line are the ordinary Clebsch-Gordan
        coefficients.  See, e.g., Campbell and Morgan (1971).

        """    	
        cg1 = CG(l1, m1, l2, m2, l3, m3)
        cg2 = CG(l1, 0.0, l2, -s, l3, -s)

        return np.sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0) / (4.0 * np.pi * (2.0 * l3 + 1))) * cg1 * cg2

    for ell in range(ell_min, ell_max + 1):
        ellp_min = max(ell_min, ell - 1)
        ellp_max = min(ell_max, ell + 1)
        for ellp in range(ellp_min, ellp_max + 1):
            for m in range(-ell, ell + 1):
                mp = round(m + 1 * sign)
                if (mp < -ellp) or (mp > ellp):
                    continue
                yield (ellp, mp, ell, m, (prefac * swsh_Y_mat_el(s, ellp, mp, 1.0, sign, ell, m)))


p_plus = functools.partial(p_plusminus, sign=+1)
p_minus = functools.partial(p_plusminus, sign=-1)
p_plus.__doc__ = p_plusminus.__doc__
p_minus.__doc__ = p_plusminus.__doc__


def momentum_flux(h):
    """Compute momentum flux from waveform

    This implements Eq. (2.11) from Ruiz et al. (2008) [0707.4654] by using
    `matrix_expectation_value` with `p_z`, `p_plus`, and `p_minus`.

    """
    from .waveform_modes import WaveformModes
    from . import h as htype
    from . import hdot as hdottype

    if not isinstance(h, WaveformModes):
        raise ValueError(
            f"Momentum flux can only be calculated from a `WaveformModes` object; this object is of type `{type(h)}`."
        )
    if h.dataType == hdottype:
        hdot = h
    elif h.dataType == htype:
        hdot = h.copy()
        hdot.dataType = hdottype
        hdot.data = h.data_dot
    else:
        raise ValueError(
            f"Input argument is expected to have data of type `h` or `hdot`; this waveform data has type `{h.data_type_string}`"
        )

    pdot = np.zeros((hdot.n_times, 3), dtype=float)

    _, p_plus_dot = matrix_expectation_value(hdot, functools.partial(p_plus, s=-2), hdot)
    _, p_minus_dot = matrix_expectation_value(hdot, functools.partial(p_minus, s=-2), hdot)
    _, p_z_dot = matrix_expectation_value(hdot, functools.partial(p_z, s=-2), hdot)

    # Convert into (x,y,z) basis
    pdot[:, 0] = 0.5 * (p_plus_dot.real + p_minus_dot.real)
    pdot[:, 1] = 0.5 * (p_plus_dot.imag - p_minus_dot.imag)
    pdot[:, 2] = p_z_dot.real

    pdot /= 16.0 * np.pi

    return pdot


@swsh_indices_to_matrix_indices
def j_z(ell_min, ell_max):
    r"""Generator for jᶻ matrix elements (for use with matrix_expectation_value)

    Matrix elements yielded are

      ⟨j,n|jᶻ|l,m⟩ = i m δⱼₗ δₙₘ

    This follows the convention for jᶻ from Ruiz et al. (2008) [0707.4654]

    """
    for ell in range(ell_min, ell_max + 1):
        for m in range(-ell, ell + 1):
            yield ell, m, ell, m, (1.0j * m)


@swsh_indices_to_matrix_indices
def j_plusminus(ell_min, ell_max, sign):
    r"""Produce the function j_plus or j_minus, based on sign.

    The conventions for these matrix elements, to agree with Ruiz et al. (2008)
    [0707.4654], should be:

      ⟨j,n|j⁺|l,m⟩ = i √[(l-m)(l+m+1)] δⱼₗ δₙₘ₊₁
      ⟨j,n|j⁻|l,m⟩ = i √[(l+m)(l-m+1)] δⱼₗ δₙₘ₋₁

    The spinsfast function `ladder_operator_coefficient(l, m)` gives
    √[(l-m)(l+m+1)].

    """
    from spherical_functions import ladder_operator_coefficient as ladder

    if (sign != 1) and (sign != -1):
        raise ValueError("sign must be either 1 or -1 in j_plusminus")

    for ell in range(ell_min, ell_max + 1):
        for m in range(-ell, ell + 1):
            mp = round(m + 1 * sign)
            if (mp < -ell) or (mp > ell):
                continue
            yield ell, mp, ell, m, (1.0j * ladder(ell, m * sign))


j_plus = functools.partial(j_plusminus, sign=+1)
j_minus = functools.partial(j_plusminus, sign=-1)
j_plus.__doc__ = j_plusminus.__doc__
j_minus.__doc__ = j_plusminus.__doc__


def angular_momentum_flux(h, hdot=None):
    """Compute angular momentum flux from waveform

    This implements Eq. (2.24) from Ruiz et al. (2008) [0707.4654] by using
    `matrix_expectation_value` with (internal) helper functions `j_z`, `j_plus`, and
    `j_minus`.

    """
    from .waveform_modes import WaveformModes
    from . import h as htype
    from . import hdot as hdottype

    if not isinstance(h, WaveformModes):
        raise ValueError(
            f"Angular momentum flux can only be calculated from a `WaveformModes` object; `h` is of type `{type(h)}`."
        )
    if (hdot is not None) and (not isinstance(hdot, WaveformModes)):
        raise ValueError(
            f"Angular momentum flux can only be calculated from a `WaveformModes` object; `hdot` is of type `{type(hdot)}`."
        )
    if h.dataType == htype:
        if hdot is None:
            hdot = h.copy()
            hdot.dataType = hdottype
            hdot.data = h.data_dot
        elif hdot.dataType != hdottype:
            raise ValueError(
                f"Input argument `hdot` is expected to have data of type `hdot`; this `hdot` waveform data has type `{h.data_type_string}`"
            )
    else:
        raise ValueError(
            f"Input argument `h` is expected to have data of type `h`; this `h` waveform data has type `{h.data_type_string}`"
        )

    jdot = np.zeros((hdot.n_times, 3), dtype=float)

    _, j_plus_dot = matrix_expectation_value(hdot, j_plus, h)
    _, j_minus_dot = matrix_expectation_value(hdot, j_minus, h)
    _, j_z_dot = matrix_expectation_value(hdot, j_z, h)

    # Convert into (x,y,z) basis
    jdot[:, 0] = 0.5 * (j_plus_dot.real + j_minus_dot.real)
    jdot[:, 1] = 0.5 * (j_plus_dot.imag - j_minus_dot.imag)
    jdot[:, 2] = j_z_dot.real

    jdot /= -16.0 * np.pi

    return jdot


def boost_flux(h, hdot=None):
    """Computes the boost flux from the waveform.

    This implements Eq. (C.1) from Flanagan & Nichols (2016) [1510.03386] for the
    boost vector field.

    Notes
    -----
    Boost flux is analytically calculated using the Wald & Zoupas formalism,
    following the calculation by Flanagan & Nichols [1510.03386].  Then, the NP
    formalism is implemented to make it computationally feasible.  The conventions
    of Ruiz et al. (2008) [0707.4654] are followed.

    The expression for boost flux is

      boost_flux(h) = (-1/32π) (
        (1/8) [⟨ð̄N|χ|ð̄h⟩ - ⟨ðN|χ|ðh⟩ + ⟨ð̄h|χ|ð̄N⟩ - ⟨ðh|χ|ðN⟩ + 6⟨N|χ|h⟩ + 6⟨h|χ|N⟩]
        - (1/4) [⟨ðN|ðχ|h⟩ + ⟨h|ð̄χ|ðN⟩]
        - (u/2) ⟨N|χ|N⟩
      )

    h and N have spin weight s=-2.  χ is an l=1, s=0 function characterizing the
    direction of the boost.  ð and ð̄ are spin raising and lowering operators.

    The matrix elements for most of the terms are identical to pᶻ, p⁺, and p⁻
    matrix elements, except that input for s is different.  For example, ⟨ð̄N|χ|ð̄h⟩
    will have s = -3.  There are two terms ⟨ðN|ðχ|h⟩ and ⟨h|ð̄χ|ðN⟩ for which matrix
    elements have been defined below.

    """
    from .waveform_modes import WaveformModes
    from . import h as htype
    from . import hdot as hdottype

    # We start by defining matrix elements that are not present already in this code.

    @swsh_indices_to_matrix_indices
    def eth_chi_z(ell_min, ell_max, s=-2):
        """Generator for the matrix element ⟨ðN|ðχ|h⟩ in the z direction

        For the z direction χ is an m = 0 function.  The matrix element is

          √[(6/4π) ((2l+1)/(2j+1))] ⟨l,-s,1,-1|j,-1-s⟩ ⟨l,m,1,0|j,m⟩

        """
        for ell in range(ell_min, ell_max + 1):
            ellp_min = max(ell_min, ell - 1)
            ellp_max = min(ell_max, ell + 1)
            for ellp in range(ellp_min, ellp_max + 1):
                cg2 = CG(ell, -s, 1, -1, ellp, -1 - s)
                prefac = np.sqrt((2.0 * ell + 1.0) / (2.0 * ellp + 1.0))
                for m in range(-ell, ell + 1):
                    if (m < -ellp) or (m > ellp):
                        continue
                    cg1 = np.sqrt(2) * CG(ell, m, 1, 0, ellp, m)
                    yield ellp, m, ell, m, (prefac * cg1 * cg2)

    @swsh_indices_to_matrix_indices
    def ethbar_chi_z(ell_min, ell_max, s=-2):
        """Generator for the matrix element ⟨h|ð̄χ|ðN⟩ in the z direction

        For the z direction χ is an m = 0 function.  The matrix element is

          √[(6/4π) ((2l+1)/(2j+1))] ⟨l,-s-1,1,1|j,-s⟩ ⟨l,m,1,0|j,m⟩

        """
        for ell in range(ell_min, ell_max + 1):
            ellp_min = max(ell_min, ell - 1)
            ellp_max = min(ell_max, ell + 1)
            for ellp in range(ellp_min, ellp_max + 1):
                cg2 = CG(ell, -s - 1, 1, 1, ellp, -s)
                prefac = np.sqrt((2.0 * ell + 1.0) / (2.0 * ellp + 1.0))
                for m in range(-ell, ell + 1):
                    if (m < -ellp) or (m > ellp):
                        continue
                    cg1 = np.sqrt(2) * CG(ell, m, 1, 0, ellp, m)
                    yield ellp, m, ell, m, (prefac * cg1 * cg2)

    # Flux components in the x and y direction can be obtained in the plusminus
    # basis.  plus and minus corresponds to the value taken by m as +1 and -1
    # respectively.  Change of basis will give the x and y component.

    @swsh_indices_to_matrix_indices
    def eth_chi_plusminus(ell_min, ell_max, sign, s=-2):
        """Compute the ⟨ðN|ðχ|h⟩ matrix element based on the sign

        Plus and minus sign corresponds to the value taken by m=±1.

        χ is an l=1 function, and the matrix element is

          √[2(2l₁+1)(2l₂+1)/(4π(2l₃+1))] ⟨l₁,-1,l₂,-s|l₃,-1-s⟩ ⟨l₁,m₁,l₂,m₂|l₃,m₃⟩

        We use `mat_el_eth_chi` to compute the matrix elements.  Note that since
        the operator has definite m=±1, we only have mp==m±1 nonvanishing in the
        matrix elements.

        """
        if (sign != 1) and (sign != -1):
            raise ValueError("sign must be either 1 or -1 in eth_N_eth_chi_h_plusminus")

        prefac = -1.0 * sign * np.sqrt(8.0 * np.pi / 3.0)

        def mat_el_eth_chi(s, l3, m3, l1, m1, l2, m2):
            cg1 = np.sqrt(2) * CG(l1, m1, l2, m2, l3, m3)
            cg2 = CG(l1, -1.0, l2, -s, l3, -1 - s)
            return np.sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0) / (4.0 * np.pi * (2.0 * l3 + 1))) * cg1 * cg2

        for ell in range(ell_min, ell_max + 1):
            ellp_min = max(ell_min, ell - 1)
            ellp_max = min(ell_max, ell + 1)
            for ellp in range(ellp_min, ellp_max + 1):
                for m in range(-ell, ell + 1):
                    mp = round(m + 1 * sign)
                    if (mp < -ellp) or (mp > ellp):
                        continue
                    yield (ellp, mp, ell, m, (prefac * mat_el_eth_chi(s, ellp, mp, 1.0, sign, ell, m)))

    eth_chi_plus = functools.partial(eth_chi_plusminus, sign=+1)
    eth_chi_minus = functools.partial(eth_chi_plusminus, sign=-1)
    eth_chi_plus.__doc__ = eth_chi_plusminus.__doc__
    eth_chi_minus.__doc__ = eth_chi_plusminus.__doc__

    @swsh_indices_to_matrix_indices
    def ethbar_chi_plusminus(ell_min, ell_max, sign, s=-2):
        """Compute the ⟨h|ðχ|ðN⟩ matrix element depending on the sign.

        Plus and minus sign corresponds to the value taken by m=±1.  Because χ is an
        l=1 function the analytical expression is

            √[2(2l₁+1)(2l₂+1)/(4π(2l₃+1))] ⟨l₁,1,l₂,-1-s|l₃,-s⟩ ⟨l₁,m₁,l₂,m₂|l₃,m₃⟩

        We use `mat_el_ethbar_chi` to compute the matrix elements.  The operator has
        definite m=±1, so we only have mp==m±1 nonvanishing in the matrix elements.

        """
        if (sign != 1) and (sign != -1):
            raise ValueError("sign must be either 1 or -1 in h_ethbar_chi_eth_N_plusminus")

        prefac = -1.0 * sign * np.sqrt(8.0 * np.pi / 3.0)

        def mat_el_ethbar_chi(s, l3, m3, l1, m1, l2, m2):
            cg1 = np.sqrt(2) * CG(l1, m1, l2, m2, l3, m3)
            cg2 = CG(l1, 1.0, l2, -1 - s, l3, -s)
            return np.sqrt((2.0 * l1 + 1.0) * (2.0 * l2 + 1.0) / (4.0 * np.pi * (2.0 * l3 + 1))) * cg1 * cg2

        for ell in range(ell_min, ell_max + 1):
            ellp_min = max(ell_min, ell - 1)
            ellp_max = min(ell_max, ell + 1)
            for ellp in range(ellp_min, ellp_max + 1):
                for m in range(-ell, ell + 1):
                    mp = round(m + 1 * sign)
                    if (mp < -ellp) or (mp > ellp):
                        continue
                    yield (ellp, mp, ell, m, (prefac * mat_el_ethbar_chi(s, ellp, mp, 1.0, sign, ell, m)))

    ethbar_chi_plus = functools.partial(ethbar_chi_plusminus, sign=+1)
    ethbar_chi_minus = functools.partial(ethbar_chi_plusminus, sign=-1)
    ethbar_chi_plus.__doc__ = ethbar_chi_plusminus.__doc__
    ethbar_chi_minus.__doc__ = ethbar_chi_plusminus.__doc__

    if not isinstance(h, WaveformModes):
        raise ValueError(
            "Boost fluxes can only be calculated from a `WaveformModes` object; "
            + "`h` is of type `{0}`.".format(type(h))
        )
    if (hdot is not None) and (not isinstance(hdot, WaveformModes)):
        raise ValueError(
            "Boost fluxes can only be calculated from a `WaveformModes` object; "
            + "`hdot` is of type `{0}`.".format(type(hdot))
        )
    if h.dataType == htype:
        if hdot is None:
            hdot = h.copy()
            hdot.dataType = hdottype
            hdot.data = h.data_dot
        elif hdot.dataType != hdottype:
            raise ValueError(
                "Input argument `hdot` is expected to have data of type `hdot`; "
                + "this `hdot` waveform data has type `{0}`".format(h.data_type_string)
            )
    else:
        raise ValueError(
            "Input argument `h` is expected to have data of type `h`; "
            + "this `h` waveform data has type `{0}`".format(h.data_type_string)
        )

    # h and h_dot are spin raised and lowered in the expression for boost flux.
    # Hence, one needs to define new waveform like object from h and hdot that
    # contains the updated data corresponding to the raising and lowering of the spin.
    # Appropriate ladder factor needs to be multiplied to each (l,m) mode.
    # eth and ethbar functions are used here for this operation.

    # eth_h --> Spin of h raised by 1; s = -1
    eth_h = h.copy()
    eth_h.data = h.eth

    # ethbar_h --> Spin of h lowered by 1; s = -3
    ethbar_h = h.copy()
    ethbar_h.data = h.ethbar

    # eth_hdot --> Spin of ḣ raised by 1; s = -1
    eth_hdot = hdot.copy()
    eth_hdot.data = hdot.eth

    # ethbar_hdot --> Spin of ḣ lowered by 1; s = -3
    ethbar_hdot = hdot.copy()
    ethbar_hdot.data = hdot.ethbar

    boost_flux = np.zeros((hdot.n_times, 3), dtype=float)

    # Each term has been computed one by taking expectation value
    # between waveform like objects (h, hdot, eth_h, ethbar_h, eth_hdot, ethbar_hdot)
    # and the matrix elements.
    # The spin of the waveform like object is taken as input for spin value s in matrix elements p_z and p_plusminus.
    # Because both ⟨a| and |b⟩ have the same spin when the matrix element is p_z and p_plusminus.

    # plus and minus terms are computed here.
    _, ethbar_hdot_chiplus_ethbar_h = matrix_expectation_value(ethbar_hdot, functools.partial(p_plus, s=-3), ethbar_h)
    _, ethbar_hdot_chiminus_ethbar_h = matrix_expectation_value(ethbar_hdot, functools.partial(p_minus, s=-3), ethbar_h)
    _, eth_hdot_chiplus_eth_h = matrix_expectation_value(eth_hdot, functools.partial(p_plus, s=-1), eth_h)
    _, eth_hdot_chiminus_eth_h = matrix_expectation_value(eth_hdot, functools.partial(p_minus, s=-1), eth_h)
    _, hdot_chiplus_h = matrix_expectation_value(hdot, functools.partial(p_plus, s=-2), h)
    _, hdot_chiminus_h = matrix_expectation_value(hdot, functools.partial(p_minus, s=-2), h)
    _, ethbar_h_chiplus_ethbar_hdot = matrix_expectation_value(ethbar_h, functools.partial(p_plus, s=-3), ethbar_hdot)
    _, ethbar_h_chiminus_ethbar_hdot = matrix_expectation_value(ethbar_h, functools.partial(p_minus, s=-3), ethbar_hdot)
    _, eth_h_chiplus_eth_hdot = matrix_expectation_value(eth_h, functools.partial(p_plus, s=-1), eth_hdot)
    _, eth_h_chiminus_eth_hdot = matrix_expectation_value(eth_h, functools.partial(p_minus, s=-1), eth_hdot)
    _, h_chiplus_hdot = matrix_expectation_value(h, functools.partial(p_plus, s=-2), hdot)
    _, h_chiminus_hdot = matrix_expectation_value(h, functools.partial(p_minus, s=-2), hdot)
    _, eth_hdot_eth_chiplus_h = matrix_expectation_value(eth_hdot, functools.partial(eth_chi_plus, s=-2), h)
    _, eth_hdot_eth_chiminus_h = matrix_expectation_value(eth_hdot, functools.partial(eth_chi_minus, s=-2), h)
    _, h_eth_chiplus_eth_hdot = matrix_expectation_value(h, functools.partial(ethbar_chi_plus, s=-2), eth_hdot)
    _, h_eth_chiminus_eth_hdot = matrix_expectation_value(h, functools.partial(ethbar_chi_minus, s=-2), eth_hdot)
    _, hdot_chiplus_hdot = matrix_expectation_value(hdot, functools.partial(p_plus, s=-2), hdot)
    _, hdot_chiminus_hdot = matrix_expectation_value(hdot, functools.partial(p_minus, s=-2), hdot)

    # Terms in the z direction are computed here.
    _, ethbar_hdot_chiz_ethbar_h = matrix_expectation_value(ethbar_hdot, functools.partial(p_z, s=-3), ethbar_h)
    _, eth_hdot_chiz_eth_h = matrix_expectation_value(eth_hdot, functools.partial(p_z, s=-1), eth_h)
    _, hdot_chiz_h = matrix_expectation_value(hdot, functools.partial(p_z, s=-2), h)
    _, ethbar_h_chiz_ethbar_hdot = matrix_expectation_value(ethbar_h, functools.partial(p_z, s=-3), ethbar_hdot)
    _, eth_h_chiz_eth_hdot = matrix_expectation_value(eth_h, functools.partial(p_z, s=-1), eth_hdot)
    _, h_chiz_hdot = matrix_expectation_value(h, functools.partial(p_z, s=-2), hdot)
    _, eth_hdot_eth_chiz_h = matrix_expectation_value(eth_hdot, functools.partial(eth_chi_z, s=-2), h)
    _, h_ethbar_chiz_eth_hdot = matrix_expectation_value(h, functools.partial(ethbar_chi_z, s=-2), eth_hdot)
    _, hdot_chiz_hdot = matrix_expectation_value(hdot, functools.partial(p_z, s=-2), hdot)

    # These terms needs to be added to get the components in a particular direction.
    # The basis will be changed from (plus, minus, z) to (x, y, z).

    boost_flux_plus = (
        (1 / 8) * (
            ethbar_hdot_chiplus_ethbar_h
            - eth_hdot_chiplus_eth_h
            + 6 * hdot_chiplus_h
            + ethbar_h_chiplus_ethbar_hdot
            - eth_h_chiplus_eth_hdot
            + 6 * h_chiplus_hdot
        )
        - (1 / 2) * np.multiply(h.t, hdot_chiplus_hdot)
        - (1 / 4) * (
            eth_hdot_eth_chiplus_h
            - h_eth_chiplus_eth_hdot
        )
    )

    boost_flux_minus = (
        (1 / 8) * (
            ethbar_hdot_chiminus_ethbar_h
            - eth_hdot_chiminus_eth_h
            + 6 * hdot_chiminus_h
            + ethbar_h_chiminus_ethbar_hdot
            - eth_h_chiminus_eth_hdot
            + 6 * h_chiminus_hdot
        )
        - (1 / 2) * np.multiply(h.t, hdot_chiminus_hdot)
        - (1 / 4) * (
            eth_hdot_eth_chiminus_h
            - h_eth_chiminus_eth_hdot
        )
    )

    # This is the component in the x direction. x = 0.5 * ( plus + minus).real
    boost_flux[:, 0] = (0.5) * (boost_flux_plus + boost_flux_minus).real

    # This is the component in the y direction. y = 0.5 * ( plus - minus).imag
    boost_flux[:, 1] = (0.5) * (boost_flux_plus - boost_flux_minus).imag

    # Component in the z direction. Only the real part of the complex value is taken into account.
    boost_flux[:, 2] = ((1 / 8) * (ethbar_hdot_chiz_ethbar_h
            - eth_hdot_chiz_eth_h
            + 6 * hdot_chiz_h
            + ethbar_h_chiz_ethbar_hdot
            - eth_h_chiz_eth_hdot
            + 6 * h_chiz_hdot)
        - (1 / 2) * np.multiply(h.t, hdot_chiz_hdot)
        + (-1 / 4) * eth_hdot_eth_chiz_h
        + (1 / 4) * h_ethbar_chiz_eth_hdot
    ).real

    # A final factor of -1/32π should be included that is present outside the integral expression of flux.
    boost_flux /= -32 * np.pi

    return boost_flux


def poincare_fluxes(h, hdot=None):
    """Compute fluxes of energy, momentum, angular momemntum, and boost

    This function will compute the time derivative 1 or 0 times (if an optional
    argument is passed) so is more efficient than separate calls to `energy_flux`,
    `momentum_flux`, `angular_momentum_flux`, and `boost_flux`.

    Parameters
    ----------
    h : WaveformModes object
        Must have type `h`

    hdot : WaveformModes object, optional [default: None]
        The time derivative of h.  If None, computed from h.  Must
        have type `hdot`.

    Returns
    -------
    edot, pdot, jdot, boost_flux : ndarray, ndarray, ndarray, ndarray

    """
    from .waveform_modes import WaveformModes
    from . import h as htype
    from . import hdot as hdottype

    if not isinstance(h, WaveformModes):
        raise ValueError(
            f"Poincare fluxes can only be calculated from a `WaveformModes` object; `h` is of type `{type(h)}`."
        )
    if (hdot is not None) and (not isinstance(hdot, WaveformModes)):
        raise ValueError(
            f"Poincare fluxes can only be calculated from a `WaveformModes` object; `hdot` is of type `{type(hdot)}`."
        )
    if h.dataType == htype:
        if hdot is None:
            hdot = h.copy()
            hdot.dataType = hdottype
            hdot.data = h.data_dot
        elif hdot.dataType != hdottype:
            raise ValueError(
                f"Input argument `hdot` is expected to have data of type `hdot`; this `hdot` waveform data has type `{h.data_type_string}`"
            )
    else:
        raise ValueError(
            f"Input argument `h` is expected to have data of type `h`; this `h` waveform data has type `{h.data_type_string}`"
        )

    return (energy_flux(hdot), momentum_flux(hdot), angular_momentum_flux(h, hdot), boost_flux(h, hdot))

