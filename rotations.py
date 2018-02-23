# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

import numpy as np
import quaternion
import spherical_functions as sf
from quaternion.numba_wrapper import njit, xrange
from .waveform_base import waveform_alterations
from .mode_calculations import corotating_frame
from . import Corotating


@waveform_alterations
def to_corotating_frame(W, R0=quaternion.one, tolerance=1e-12):
    W.rotate_decomposition_basis(corotating_frame(W, R0=R0, tolerance=1e-12))
    W._append_history('{0}.to_corotating_frame({1}, {2})'.format(W, R0, tolerance))
    W.frameType = Corotating
    return W


@waveform_alterations
def rotate_physical_system(W, R_phys):
    """Rotate a Waveform in place

    This just rotates the decomposition basis by the inverse of the input
    rotor(s).  See `rotate_decomposition_basis`.

    For more information on the analytical details, see
    http://moble.github.io/spherical_functions/SWSHs.html#rotating-swshs

    """
    W = rotate_decomposition_basis(W, ~R_phys)
    W._append_history('{0}.rotate_physical_system(...)'.format(W))
    return W  # Probably no return, but just in case...


@waveform_alterations
def rotate_decomposition_basis(W, R_basis):
    """Rotate a Waveform in place

    This function takes a Waveform object `W` and either a quaternion
    or array of quaternions `R_basis`.  It applies that rotation to
    the decomposition basis of the modes in the Waveform.  The change
    in basis is also recorded in the Waveform's `frame` data.

    For more information on the analytical details, see
    http://moble.github.io/spherical_functions/SWSHs.html#rotating-swshs

    """
    # This will be used in the jitted functions below to store the
    # Wigner D matrix at each time step
    D = np.empty((sf.WignerD._total_size_D_matrices(W.ell_min, W.ell_max),), dtype=complex)

    if (isinstance(R_basis, (list, np.ndarray)) and len(R_basis) == 1):
        R_basis = R_basis[0]

    if (isinstance(R_basis, (list, np.ndarray))):
        if (isinstance(R_basis, np.ndarray) and R_basis.ndim != 1):
            raise ValueError("Input dimension mismatch.  R_basis.shape={1}".format(R_basis.shape))
        if (W.n_times != len(R_basis)):
            raise ValueError(
                "Input dimension mismatch.  (W.n_times={0}) != (len(R_basis)={1})".format(W.n_times, len(R_basis)))
        _rotate_decomposition_basis_by_series(W.data, quaternion.as_spinor_array(R_basis), W.ell_min, W.ell_max, D)

        # Update the frame data, using right-multiplication
        if (W.frame.size):
            if (W.frame.shape[0] == 1):
                # Numpy can't currently multiply one element times an array
                W.frame = np.array([W.frame * R for R in R_basis])
            else:
                W.frame = W.frame * R_basis
        else:
            W.frame = np.copy(R_basis)

    # We can't just use an `else` here because we need to process the
    # case where the input was an iterable of length 1, which we've
    # now changed to just a single quaternion.
    if (isinstance(R_basis, np.quaternion)):
        sf._Wigner_D_matrices(R_basis.a, R_basis.b, W.ell_min, W.ell_max, D)
        tmp = np.empty((2 * W.ell_max + 1,), dtype=complex)
        _rotate_decomposition_basis_by_constant(W.data, W.ell_min, W.ell_max, D, tmp)

        # Update the frame data, using right-multiplication
        if (W.frame.size):
            W.frame = W.frame * R_basis
        else:
            W.frame = np.array([R_basis])

    opts = np.get_printoptions()
    np.set_printoptions(threshold=6)

    W.__history_depth__ -= 1
    W._append_history('{0}.rotate_decomposition_basis({1})'.format(W, R_basis))
    np.set_printoptions(**opts)

    return W


@njit('void(c16[:,:], i8, i8, c16[:], c16[:])')
def _rotate_decomposition_basis_by_constant(data, ell_min, ell_max, D, tmp):
    """Rotate data by the same rotor at each point in time

    `D` is the Wigner D matrix for all the ell values.

    `tmp` is just a workspace used as temporary storage to hold the
    results for each item of data during the sum.

    """
    for i_t in xrange(data.shape[0]):
        for ell in xrange(ell_min, ell_max + 1):
            i_data = ell ** 2 - ell_min ** 2
            i_D = sf._linear_matrix_offset(ell, ell_min)

            for i_m in xrange(2 * ell + 1):
                tmp[i_m] = 0j
            for i_mp in xrange(2 * ell + 1):
                for i_m in xrange(2 * ell + 1):
                    tmp[i_m] += data[i_t, i_data + i_mp] * D[i_D + (2 * ell + 1) * i_mp + i_m]
            for i_m in xrange(2 * ell + 1):
                data[i_t, i_data + i_m] = tmp[i_m]


@njit('void(c16[:,:], c16[:,:], i8, i8, c16[:])')
def _rotate_decomposition_basis_by_series(data, R_basis, ell_min, ell_max, D):
    """Rotate data by a different rotor at each point in time

    `D` is just a workspace, which holds the Wigner D matrices.
    During the summation, it is also used as temporary storage to hold
    the results for each item of data, where the first row in the
    matrix is overwritten with the new sums.

    """
    for i_t in xrange(data.shape[0]):
        sf._Wigner_D_matrices(R_basis[i_t, 0], R_basis[i_t, 1], ell_min, ell_max, D)
        for ell in xrange(ell_min, ell_max + 1):
            i_data = ell ** 2 - ell_min ** 2
            i_D = sf._linear_matrix_offset(ell, ell_min)

            for i_m in xrange(2 * ell + 1):
                new_data_mp = 0j
                for i_mp in xrange(2 * ell + 1):
                    new_data_mp += data[i_t, i_data + i_mp] * D[i_D + i_m + (2 * ell + 1) * i_mp]
                D[i_D + i_m] = new_data_mp
            for i_m in xrange(2 * ell + 1):
                data[i_t, i_data + i_m] = D[i_D + i_m]
