# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

from math import sqrt

import numpy as np
import quaternion

from spherical_functions import ladder_operator_coefficient as ladder
from spherical_functions import clebsch_gordan as CG
from quaternion.numba_wrapper import jit, njit, xrange


@njit("void(c16[:,:], c16[:,:], i8[:,:], f8[:,:])")
def _LdtVector(data, datadot, lm, Ldt):
    """Helper function for the LdtVector function"""
    # Big, bad, ugly, obvious way to do the calculation
    # =================================================
    # L+ = Lx + i Ly      Lx =    (L+ + L-) / 2
    # L- = Lx - i Ly      Ly = -i (L+ - L-) / 2

    for i_mode in xrange(lm.shape[0]):
        L = lm[i_mode, 0]
        M = lm[i_mode, 1]
        for i_time in xrange(data.shape[0]):
            # Compute first in (+,-,z) basis
            Lp = (
                np.conjugate(data[i_time, i_mode + 1])
                * datadot[i_time, i_mode]
                * ladder(L, M)
                if M + 1 <= L
                else 0.0 + 0.0j
            )
            Lm = (
                np.conjugate(data[i_time, i_mode - 1])
                * datadot[i_time, i_mode]
                * ladder(L, -M)
                if M - 1 >= -L
                else 0.0 + 0.0j
            )
            Lz = np.conjugate(data[i_time, i_mode]) * datadot[i_time, i_mode] * M

            # Convert into (x,y,z) basis
            Ldt[i_time, 0] += 0.5 * (Lp.imag + Lm.imag)
            Ldt[i_time, 1] += -0.5 * (Lp.real - Lm.real)
            Ldt[i_time, 2] += Lz.imag
    return


def LdtVector(W):
    r"""Calculate the <Ldt> quantity with respect to the modes

    The vector is given in the (possibly rotating) mode frame (X,Y,Z),
    rather than the inertial frame (x,y,z).

    <Ldt>^{a} = \sum_{\ell,m,m'} \bar{f}^{\ell,m'} < \ell,m' | L_a | \ell,m > (df/dt)^{\ell,m}

    """
    Ldt = np.zeros((W.n_times, 3), dtype=float)
    _LdtVector(W.data, W.data_dot, W.LM, Ldt)
    return Ldt


@njit("void(c16[:,:], c16[:,:], i8[:,:], c16[:,:])")
def _LVector(data1, data2, lm, Lvec):
    """Helper function for the LVector function"""
    # Big, bad, ugly, obvious way to do the calculation
    # =================================================
    # L+ = Lx + i Ly      Lx =    (L+ + L-) / 2
    # L- = Lx - i Ly      Ly = -i (L+ - L-) / 2

    for i_mode in xrange(lm.shape[0]):
        L = lm[i_mode, 0]
        M = lm[i_mode, 1]
        for i_time in xrange(data1.shape[0]):
            # Compute first in (+,-,z) basis
            Lp = (
                np.conjugate(data1[i_time, i_mode + 1])
                * data2[i_time, i_mode]
                * ladder(L, M)
                if M + 1 <= L
                else 0.0 + 0.0j
            )
            Lm = (
                np.conjugate(data1[i_time, i_mode - 1])
                * data2[i_time, i_mode]
                * ladder(L, -M)
                if M - 1 >= -L
                else 0.0 + 0.0j
            )
            Lz = np.conjugate(data1[i_time, i_mode]) * data2[i_time, i_mode] * M

            # Convert into (x,y,z) basis
            Lvec[i_time, 0] += 0.5 * (Lp + Lm)
            Lvec[i_time, 1] += -0.5j * (Lp - Lm)
            Lvec[i_time, 2] += Lz
    return


def LVector(W1, W2):
    r"""Calculate the <L> quantity with respect to the modes

    The vector is given in the (possibly rotating) mode frame (X,Y,Z),
    rather than the inertial frame (x,y,z).

    <L>^{a} = \sum_{\ell,m,m'} \bar{f}^{\ell,m'} < \ell,m' | L_a | \ell,m > g^{\ell,m}

    """
    L = np.zeros((W1.n_times, 3), dtype=complex)
    _LVector(W1.data, W2.data, W1.LM, L)
    return L


@njit("void(c16[:,:], c16[:,:], i8[:,:], c16[:,:,:])")
def _LLComparisonMatrix(data1, data2, lm, LL):
    """Helper function for the LLComparisonMatrix function"""
    # Big, bad, ugly, obvious way to do the calculation
    # =================================================
    # L+ = Lx + i Ly      Lx =    (L+ + L-) / 2     Im(Lx) =  ( Im(L+) + Im(L-) ) / 2
    # L- = Lx - i Ly      Ly = -i (L+ - L-) / 2     Im(Ly) = -( Re(L+) - Re(L-) ) / 2
    # Lz = Lz             Lz = Lz                   Im(Lz) = Im(Lz)
    # LxLx =   (L+ + L-)(L+ + L-) / 4
    # LxLy = -i(L+ + L-)(L+ - L-) / 4
    # LxLz =   (L+ + L-)(  Lz   ) / 2
    # LyLx = -i(L+ - L-)(L+ + L-) / 4
    # LyLy =  -(L+ - L-)(L+ - L-) / 4
    # LyLz = -i(L+ - L-)(  Lz   ) / 2
    # LzLx =   (  Lz   )(L+ + L-) / 2
    # LzLy = -i(  Lz   )(L+ - L-) / 2
    # LzLz =   (  Lz   )(  Lz   )

    for i_mode in xrange(lm.shape[0]):
        L = lm[i_mode, 0]
        M = lm[i_mode, 1]
        for i_time in xrange(data1.shape[0]):
            # Compute first in (+,-,z) basis
            LpLp = (
                np.conjugate(data1[i_time, i_mode + 2])
                * data2[i_time, i_mode]
                * (ladder(L, M + 1) * ladder(L, M))
                if M + 2 <= L
                else 0.0 + 0.0j
            )
            LpLm = (
                np.conjugate(data1[i_time, i_mode])
                * data2[i_time, i_mode]
                * (ladder(L, M - 1) * ladder(L, -M))
                if M - 1 >= -L
                else 0.0 + 0.0j
            )
            LmLp = (
                np.conjugate(data1[i_time, i_mode])
                * data2[i_time, i_mode]
                * (ladder(L, -(M + 1)) * ladder(L, M))
                if M + 1 <= L
                else 0.0 + 0.0j
            )
            LmLm = (
                np.conjugate(data1[i_time, i_mode - 2])
                * data2[i_time, i_mode]
                * (ladder(L, -(M - 1)) * ladder(L, -M))
                if M - 2 >= -L
                else 0.0 + 0.0j
            )
            LpLz = (
                np.conjugate(data1[i_time, i_mode + 1])
                * data2[i_time, i_mode]
                * (ladder(L, M) * M)
                if M + 1 <= L
                else 0.0 + 0.0j
            )
            LzLp = (
                np.conjugate(data1[i_time, i_mode + 1])
                * data2[i_time, i_mode]
                * ((M + 1) * ladder(L, M))
                if M + 1 <= L
                else 0.0 + 0.0j
            )
            LmLz = (
                np.conjugate(data1[i_time, i_mode - 1])
                * data2[i_time, i_mode]
                * (ladder(L, -M) * M)
                if M - 1 >= -L
                else 0.0 + 0.0j
            )
            LzLm = (
                np.conjugate(data1[i_time, i_mode - 1])
                * data2[i_time, i_mode]
                * ((M - 1) * ladder(L, -M))
                if M - 1 >= -L
                else 0.0 + 0.0j
            )
            LzLz = np.conjugate(data1[i_time, i_mode]) * data2[i_time, i_mode] * M ** 2

            # Convert into (x,y,z) basis
            LL[i_time, 0, 0] += 0.25 * (LpLp + LmLm + LmLp + LpLm)
            LL[i_time, 0, 1] += -0.25j * (LpLp - LmLm + LmLp - LpLm)
            LL[i_time, 0, 2] += 0.5 * (LpLz + LmLz)
            LL[i_time, 1, 0] += -0.25j * (LpLp - LmLp + LpLm - LmLm)
            LL[i_time, 1, 1] += -0.25 * (LpLp - LmLp - LpLm + LmLm)
            LL[i_time, 1, 1] += -0.5j * (LpLz - LmLz)
            LL[i_time, 2, 0] += 0.5 * (LzLp + LzLm)
            LL[i_time, 2, 1] += -0.5j * (LzLp - LzLm)
            LL[i_time, 2, 2] += LzLz

            # # Symmetrize
            # LL[i_time,0,0] += ( LxLx ).real
            # LL[i_time,0,1] += ( LxLy + LyLx ).real/2.0
            # LL[i_time,0,2] += ( LxLz + LzLx ).real/2.0
            # LL[i_time,1,0] += ( LyLx + LxLy ).real/2.0
            # LL[i_time,1,1] += ( LyLy ).real
            # LL[i_time,1,2] += ( LyLz + LzLy ).real/2.0
            # LL[i_time,2,0] += ( LzLx + LxLz ).real/2.0
            # LL[i_time,2,1] += ( LzLy + LyLz ).real/2.0
            # LL[i_time,2,2] += ( LzLz ).real
    return


def LLComparisonMatrix(W1, W2):
    r"""Calculate the <LL> quantity with respect to the modes of two Waveforms

    The matrix is given in the (possibly rotating) mode frame (X,Y,Z),
    rather than the inertial frame (x,y,z).

    <LL>^{ab} = \sum_{\ell,m,m'} \bar{f}^{\ell,m'} < \ell,m' | L_a L_b | \ell,m > g^{\ell,m}

    """
    LL = np.zeros((W1.n_times, 3, 3), dtype=complex)
    _LLComparisonMatrix(W1.data, W2.data, W1.LM, LL)
    return LL


@njit("void(c16[:,:], i8[:,:], f8[:,:,:])")
def _LLMatrix(data, lm, LL):
    """Helper function for the LLMatrix function"""
    # Big, bad, ugly, obvious way to do the calculation
    # =================================================
    # L+ = Lx + i Ly      Lx =    (L+ + L-) / 2     Im(Lx) =  ( Im(L+) + Im(L-) ) / 2
    # L- = Lx - i Ly      Ly = -i (L+ - L-) / 2     Im(Ly) = -( Re(L+) - Re(L-) ) / 2
    # Lz = Lz             Lz = Lz                   Im(Lz) = Im(Lz)
    # LxLx =   (L+ + L-)(L+ + L-) / 4
    # LxLy = -i(L+ + L-)(L+ - L-) / 4
    # LxLz =   (L+ + L-)(  Lz   ) / 2
    # LyLx = -i(L+ - L-)(L+ + L-) / 4
    # LyLy =  -(L+ - L-)(L+ - L-) / 4
    # LyLz = -i(L+ - L-)(  Lz   ) / 2
    # LzLx =   (  Lz   )(L+ + L-) / 2
    # LzLy = -i(  Lz   )(L+ - L-) / 2
    # LzLz =   (  Lz   )(  Lz   )

    for i_mode in xrange(lm.shape[0]):
        L = lm[i_mode, 0]
        M = lm[i_mode, 1]
        for i_time in xrange(data.shape[0]):
            # Compute first in (+,-,z) basis
            LpLp = (
                np.conjugate(data[i_time, i_mode + 2])
                * data[i_time, i_mode]
                * (ladder(L, M + 1) * ladder(L, M))
                if M + 2 <= L
                else 0.0 + 0.0j
            )
            LpLm = (
                np.conjugate(data[i_time, i_mode])
                * data[i_time, i_mode]
                * (ladder(L, M - 1) * ladder(L, -M))
                if M - 1 >= -L
                else 0.0 + 0.0j
            )
            LmLp = (
                np.conjugate(data[i_time, i_mode])
                * data[i_time, i_mode]
                * (ladder(L, -(M + 1)) * ladder(L, M))
                if M + 1 <= L
                else 0.0 + 0.0j
            )
            LmLm = (
                np.conjugate(data[i_time, i_mode - 2])
                * data[i_time, i_mode]
                * (ladder(L, -(M - 1)) * ladder(L, -M))
                if M - 2 >= -L
                else 0.0 + 0.0j
            )
            LpLz = (
                np.conjugate(data[i_time, i_mode + 1])
                * data[i_time, i_mode]
                * (ladder(L, M) * M)
                if M + 1 <= L
                else 0.0 + 0.0j
            )
            LzLp = (
                np.conjugate(data[i_time, i_mode + 1])
                * data[i_time, i_mode]
                * ((M + 1) * ladder(L, M))
                if M + 1 <= L
                else 0.0 + 0.0j
            )
            LmLz = (
                np.conjugate(data[i_time, i_mode - 1])
                * data[i_time, i_mode]
                * (ladder(L, -M) * M)
                if M - 1 >= -L
                else 0.0 + 0.0j
            )
            LzLm = (
                np.conjugate(data[i_time, i_mode - 1])
                * data[i_time, i_mode]
                * ((M - 1) * ladder(L, -M))
                if M - 1 >= -L
                else 0.0 + 0.0j
            )
            LzLz = np.conjugate(data[i_time, i_mode]) * data[i_time, i_mode] * M ** 2

            # Convert into (x,y,z) basis
            LxLx = 0.25 * (LpLp + LmLm + LmLp + LpLm)
            LxLy = -0.25j * (LpLp - LmLm + LmLp - LpLm)
            LxLz = 0.5 * (LpLz + LmLz)
            LyLx = -0.25j * (LpLp - LmLp + LpLm - LmLm)
            LyLy = -0.25 * (LpLp - LmLp - LpLm + LmLm)
            LyLz = -0.5j * (LpLz - LmLz)
            LzLx = 0.5 * (LzLp + LzLm)
            LzLy = -0.5j * (LzLp - LzLm)
            # LzLz = (LzLz)

            # Symmetrize
            LL[i_time, 0, 0] += LxLx.real
            LL[i_time, 0, 1] += (LxLy + LyLx).real / 2.0
            LL[i_time, 0, 2] += (LxLz + LzLx).real / 2.0
            LL[i_time, 1, 0] += (LyLx + LxLy).real / 2.0
            LL[i_time, 1, 1] += LyLy.real
            LL[i_time, 1, 2] += (LyLz + LzLy).real / 2.0
            LL[i_time, 2, 0] += (LzLx + LxLz).real / 2.0
            LL[i_time, 2, 1] += (LzLy + LyLz).real / 2.0
            LL[i_time, 2, 2] += LzLz.real
    return


def LLMatrix(W):
    r"""Calculate the <LL> quantity with respect to the modes

    The matrix is given in the (possibly rotating) mode frame (X,Y,Z),
    rather than the inertial frame (x,y,z).

    This quantity is as prescribed by O'Shaughnessy et al.
    <http://arxiv.org/abs/1109.5224>, except that no normalization is
    imposed, and this operates on whatever type of data is input.

    <LL>^{ab} = Re \{ \sum_{\ell,m,m'} \bar{f}^{\ell,m'} < \ell,m' | L_a L_b | \ell,m > f^{\ell,m} \}

    """
    LL = np.zeros((W.n_times, 3, 3), dtype=float)
    _LLMatrix(W.data, W.LM, LL)
    return LL


@njit("void(f8[:,:], f8[:], i8)")
def _LLDominantEigenvector(dpa, dpa_i, i_index):
    """Jitted helper function for LLDominantEigenvector"""
    # Make the initial direction closer to RoughInitialEllDirection than not
    if (
        dpa_i[0] * dpa[i_index, 0]
        + dpa_i[1] * dpa[i_index, 1]
        + dpa_i[2] * dpa[i_index, 2]
    ) < 0.0:
        dpa[i_index, 0] *= -1
        dpa[i_index, 1] *= -1
        dpa[i_index, 2] *= -1
    # Now, go through and make the vectors reasonably continuous.
    d = -1
    LastNorm = sqrt(dpa[i_index, 0] ** 2 + dpa[i_index, 1] ** 2 + dpa[i_index, 2] ** 2)
    for i in xrange(i_index - 1, -1, -1):
        Norm = dpa[i, 0] ** 2 + dpa[i, 1] ** 2 + dpa[i, 2] ** 2
        dNorm = (
            (dpa[i, 0] - dpa[i - d, 0]) ** 2
            + (dpa[i, 1] - dpa[i - d, 1]) ** 2
            + (dpa[i, 2] - dpa[i - d, 2]) ** 2
        )
        if dNorm > Norm:
            dpa[i, 0] *= -1
            dpa[i, 1] *= -1
            dpa[i, 2] *= -1
        # While we're here, let's just normalize that last one
        if LastNorm != 0.0 and LastNorm != 1.0:
            dpa[i - d, 0] /= LastNorm
            dpa[i - d, 1] /= LastNorm
            dpa[i - d, 2] /= LastNorm
        LastNorm = sqrt(Norm)
    if LastNorm != 0.0 and LastNorm != 1.0:
        dpa[0, 0] /= LastNorm
        dpa[0, 1] /= LastNorm
        dpa[0, 2] /= LastNorm
    d = 1
    LastNorm = sqrt(dpa[i_index, 0] ** 2 + dpa[i_index, 1] ** 2 + dpa[i_index, 2] ** 2)
    for i in xrange(i_index + 1, dpa.shape[0]):
        Norm = dpa[i, 0] ** 2 + dpa[i, 1] ** 2 + dpa[i, 2] ** 2
        dNorm = (
            (dpa[i, 0] - dpa[i - d, 0]) ** 2
            + (dpa[i, 1] - dpa[i - d, 1]) ** 2
            + (dpa[i, 2] - dpa[i - d, 2]) ** 2
        )
        if dNorm > Norm:
            dpa[i, 0] *= -1
            dpa[i, 1] *= -1
            dpa[i, 2] *= -1
        # While we're here, let's just normalize that last one
        if LastNorm != 0.0 and LastNorm != 1.0:
            dpa[i - d, 0] /= LastNorm
            dpa[i - d, 1] /= LastNorm
            dpa[i - d, 2] /= LastNorm
        LastNorm = sqrt(Norm)
    if LastNorm != 0.0 and LastNorm != 1.0:
        dpa[-1, 0] /= LastNorm
        dpa[-1, 1] /= LastNorm
        dpa[-1, 2] /= LastNorm
    return


def LLDominantEigenvector(
    W, RoughDirection=np.array([0.0, 0.0, 1.0]), RoughDirectionIndex=0
):
    """Calculate the principal axis of the LL matrix

    \param Lmodes L modes to evaluate (optional)
    \param RoughDirection Vague guess about the preferred initial (optional)

    If Lmodes is empty (default), all L modes are used.  Setting
    Lmodes to [2] or [2,3,4], for example, restricts the range of
    the sum.

    Ell is the direction of the angular velocity for a PN system, so
    some rough guess about that direction allows us to choose the
    direction of the eigenvectors output by this function to be more
    parallel than anti-parallel to that direction.  The default is
    to simply choose the z axis, since this is most often the
    correct choice anyway.

    The vector is given in the (possibly rotating) mode frame
    (X,Y,Z), rather than the inertial frame (x,y,z).

    """
    # Calculate the LL matrix at each instant
    LL = np.zeros((W.n_times, 3, 3), dtype=float)
    _LLMatrix(W.data, W.LM, LL)

    # This is the eigensystem
    eigenvals, eigenvecs = np.linalg.eigh(LL)

    # Now we find and normalize the dominant principal axis at each
    # moment, made continuous
    dpa = eigenvecs[:, :, 2]  # `eigh` always returns eigenvals in *increasing* order
    _LLDominantEigenvector(dpa, RoughDirection, RoughDirectionIndex)

    return dpa


# @jit
def angular_velocity(W, include_frame_velocity=False):
    """Angular velocity of Waveform

    This function calculates the angular velocity of a Waveform object from
    its modes.  This was introduced in Sec. II of "Angular velocity of
    gravitational radiation and the corotating frame"
    <http://arxiv.org/abs/1302.2919>.  Essentially, this is the angular
    velocity of the rotating frame in which the time dependence of the modes
    is minimized.

    The vector is given in the (possibly rotating) mode frame (X,Y,Z),
    which is not necessarily equal to the inertial frame (x,y,z).

    """

    # Calculate the <Ldt> vector and <LL> matrix at each instant
    l = W.LdtVector()
    ll = W.LLMatrix()

    # Solve <Ldt> = - <LL> . omega
    omega = -np.linalg.solve(ll, l)

    if include_frame_velocity and len(W.frame) == W.n_times:
        from quaternion import derivative, as_quat_array, as_float_array

        Rdot = as_quat_array(derivative(as_float_array(W.frame), W.t))
        omega += as_float_array(2 * Rdot * W.frame.conjugate())[:, 1:]

    return omega


# def corotating_frame(W, Ri=quaternion.one, ti=0.0, tolerance=1e-12):
def corotating_frame(
    W, R0=quaternion.one, tolerance=1e-12, z_alignment_region=None, return_omega=False
):
    """Return rotor taking current mode frame into corotating frame

    This function simply evaluates the angular velocity of the waveform, and
    then integrates it to find the corotating frame itself.  This frame is
    defined to be the frame in which the time-dependence of the waveform is
    minimized --- at least, to the extent possible with a time-dependent
    rotation.  This frame is only unique up to a single overall rotation, which
    is passed in as an optional argument to this function.

    Parameters
    ----------
    W: Waveform
    R0: quaternion [defaults to 1]
        Value of the output rotation at the first output instant
    tolerance: float [defaults to 1e-12]
        Absolute tolerance used in integration
    z_alignment_region: None or 2-tuple of floats [defaults to None]
        If not None, the dominant eigenvector of the <LL> matrix is aligned with the z axis,
        averaging over this portion of the data.  The first and second elements of the input are
        considered fractions of the inspiral at which to begin and end the average.  For example,
        (0.1, 0.9) would lead to starting 10% of the time from the first time step to the max norm
        time, and ending at 90% of that time.
    return_omega: bool [defaults to False]
        If True, return a 2-tuple consisting of the frame (the usual returned object) and the
        angular-velocity data.  That is frequently also needed, so this is just a more efficient
        way of getting the data.

    """
    from quaternion.quaternion_time_series import integrate_angular_velocity, squad

    omega = angular_velocity(W)
    t, frame = integrate_angular_velocity(
        (W.t, omega), t0=W.t[0], t1=W.t[-1], R0=R0, tolerance=tolerance
    )
    if z_alignment_region is None:
        correction_rotor = quaternion.one
    else:
        initial_time = W.t[0]
        inspiral_time = W.max_norm_time() - initial_time
        t1 = initial_time + z_alignment_region[0] * inspiral_time
        t2 = initial_time + z_alignment_region[1] * inspiral_time
        i1 = np.argmin(np.abs(W.t - t1))
        i2 = np.argmin(np.abs(W.t - t2))
        R = frame[i1:i2]
        i1m = max(0, i1 - 10)
        i1p = i1m + 21
        RoughDirection = W[i1m:i1p].angular_velocity()[10]
        Vhat = W[i1:i2].LLDominantEigenvector(
            RoughDirection=RoughDirection, RoughDirectionIndex=0
        )
        Vhat_corot = np.array(
            [
                (Ri.conjugate() * quaternion.quaternion(*Vhati) * Ri).vec
                for Ri, Vhati in zip(R, Vhat)
            ]
        )
        Vhat_corot_mean = quaternion.quaternion(
            *np.mean(Vhat_corot, axis=0)
        ).normalized()
        # print(i1, i2, i1m, i1p, RoughDirection, Vhat[0], Vhat_corot[0], Vhat_corot_mean)
        correction_rotor = np.sqrt_of_rotor(-quaternion.z * Vhat_corot_mean).inverse()
    # R = squad(R, t, W.t)
    if return_omega:
        return (frame * correction_rotor, omega)
    else:
        return frame * correction_rotor


def inner_product(t, abar, b, axis=None, apply_conjugate=False):
    """Perform a time-domain complex inner product between two waveforms <a, b>.

    This is implemented using spline interpolation, calling
    quaternion.calculus.spline_definite_integral

    Parameters
    ----------
    t : array_like
        Time samples for waveforms abar and b.
    abar : array_like
        The conjugate of the 'a' waveform in the inner product (or
        simply a, if apply_conjugate=True is been passed).  Must have the
        same shape as b.
    b : array_like
        The 'b' waveform in the inner product.  Must have the same
        shape as a.
    axis : int, optional
        When abar and b are multidimensional, the inner product will
        be computed over this axis and the result will be one
        dimension lower.  Default is None, will be inferred by
        `spline_definite_integral`.
    apply_conjugate : bool, optional
        Whether or not to conjugate the abar argument before
        computing.  True means inner_product will perform the conjugation
        for you.  Default is False, meaning you have already
        performed the conjugation.

    Returns
    -------
    inner_product : ndarray
        The integral along 'axis'
    """

    from quaternion.calculus import spline_definite_integral as sdi

    if not apply_conjugate:
        integrand = abar*b
    else:
        integrand = np.conjugate(abar)*b

    return sdi(integrand, t, axis=axis)

@njit
def sparse_expectation_value(abar, rows, columns, values, b):
    """Low-level numba-friendly helper function for the main
    calculation of `matrix_expectation_value`.

    Parameters
    ----------
    abar : ndarray
        The data for <a| (after complex conjugation was applied).

    rows, columns : list
        Lists of same length, containing indices into (respectively)
        abar and b.

    values : ndarray
        Should have shape (n,) where n is the same length as rows,
        columns.  These values are the matrix elements themselves to
        be summed.

    b : ndarray
        The data for |b>. Must have the same shape as abar.

    Returns
    -------
    expectation_value : ndarray

    """
    n_times = abar.shape[0]
    n_elements = len(rows)
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

    M : evaluable
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

    rows      = []
    columns   = []
    M_values  = []

    for ellp, mp, ell, m, M_el in M(ell_min, ell_max):
        rows.append(LM_index(ellp, mp, ell_min))
        columns.append(LM_index(ell , m , ell_min))
        M_values.append(M_el)

    return (times,
            sparse_expectation_value(Abar_data, rows, columns, M_values, B_data))

def energy_flux(h):
    """Compute energy flux from waveform

    This implements Eq. (2.8) from Ruiz+ (2008) [0707.4654].
    """
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

def _swsh_Y_mat_el(s, l3, m3, l1, m1, l2, m2):
    """Compute a matrix element treating Y_{\ell, m} as a linear operator

    From the rules for the Wigner D matrices, we get the result that
    <s, l3, m3 | Y_{l1, m1} | s, l2, m2 > =
      \sqrt{ \frac{(2*l1+1)(2*l2+1)}{4*\pi*(2*l3+1)} } *
      < l1, m1, l2, m2 | l3, m3 > < l1, 0, l2, −s | l3, −s >
    where the terms on the last line are the ordinary Clebsch-Gordan coefficients.
    See e.g. Campbell and Morgan (1971).
    """

    cg1 = CG(l1, m1, l2, m2, l3, m3)
    cg2 = CG(l1, 0., l2, -s, l3, -s)

    return np.sqrt( (2.*l1 + 1.) * (2.*l2 + 1.) / (4. * np.pi * (2.*l3 + 1)) ) * cg1 * cg2

def momentum_flux(h):
    """Compute momentum flux from waveform

    This implements Eq. (2.11) from Ruiz+ (2008) [0707.4654] by using
    `matrix_expectation_value` with (internal) helper functions
    `p_z_M_for_s_minus_2`, `p_plus_M_for_s_minus_2`, and
    `p_minus_M_for_s_minus_2`.
    """

    ##############################
    # First, we need to define some helper functions to go into
    # `matrix_expectation_value`

    def p_z_M_for_s_minus_2(ell_min, ell_max):
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

        s = -2

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

    def _make_p_plus_minus_for_s_minus_2(sign):
        u"""Produce the function p_plus_M_for_s_minus_2 or p_minus_M_for_s_minus_2, based on sign.

        This function is specific to the case where waveforms have s=-2.

        p^+ = -\sqrt{8 \pi / 3} Y_{1,+1} = \sin\theta e^{+i\phi}
        p^- = +\sqrt{8 \pi / 3} Y_{1,-1} = \sin\theta e^{-i\phi}
        This is what Ruiz+ (2008) [0707.4654] calls "l^±", which is a bad name.

        We use `_swsh_Y_mat_el` to compute the matrix elements.
        Notice that since the operator has definite m = ±1, we only have
        mp == m ± 1 nonvanishing in the matrix elements.
        """

        sign = float(sign)

        if (sign != 1.) and (sign != -1.):
            raise ValueError("sign must be either 1. or -1. in make_j_plus_minus")

        s = -2
        prefac = -1. * sign * np.sqrt( 8. * np.pi / 3. )

        def p_pm_M(ell_min, ell_max):
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
                                _swsh_Y_mat_el(s, ellp, mp, 1., sign, ell, m)))

        return p_pm_M

    p_plus_M_for_s_minus_2  = _make_p_plus_minus_for_s_minus_2(1.)
    p_minus_M_for_s_minus_2 = _make_p_plus_minus_for_s_minus_2(-1.)

    ##############################
    # Now we can use `matrix_expectation_value` to evaluate the
    # momentum flux. First some plumbing.

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

    ##############################
    # Finally do the calculation

    pdot = np.zeros((hdot.n_times, 3), dtype=float)

    _, p_plus_dot  = matrix_expectation_value( hdot, p_plus_M_for_s_minus_2,  hdot )
    _, p_minus_dot = matrix_expectation_value( hdot, p_minus_M_for_s_minus_2, hdot )
    _, p_z_dot = matrix_expectation_value( hdot, p_z_M_for_s_minus_2, hdot )

    # Convert into (x,y,z) basis
    pdot[:,0] = 0.5 * ( p_plus_dot.real + p_minus_dot.real )
    pdot[:,1] = 0.5 * ( p_plus_dot.imag - p_minus_dot.imag )
    pdot[:,2] = p_z_dot.real

    pdot /= 16.*np.pi

    return pdot

def angular_momentum_flux(h, hdot=None):
    """Compute angular momentum flux from waveform

    This implements Eq. (2.24) from Ruiz+ (2008) [0707.4654] by using
    `matrix_expectation_value` with (internal) helper functions
    `j_z_M`, `j_plus_M`, and `j_minus_M`.
    """

    ##############################
    # First, we need to define some helper functions to go into
    # `matrix_expectation_value`

    def j_z_M(ell_min, ell_max):
        """Generator for j^z matrix elements (for use with matrix_expectation_value)

        Matrix elements yielded are
        <ellp, mp|j^z|ell, m> = 1.j * m \delta_{ellp ell} \delta_{mp, m}
        This follows the convention for j^z from Ruiz+ (2008) [0707.4654]
        """

        for ell in range(ell_min, ell_max+1):
            for m in range(-ell, ell+1):
                yield ell, m, ell, m, (1.j * m)

    def _make_j_plus_minus(sign):
        """Produce the function j_plus_M or j_minus_M, based on sign.

        The conventions for these matrix elements, to agree with Ruiz+
        (2008) [0707.4654], should be:
        <ellp, mp|j^+|ell, m> = 1.j * np.sqrt((l-m)(l+m+1)) \delta{ellp ell} \delta{mp (m+1)}
        <ellp, mp|j^-|ell, m> = 1.j * np.sqrt((l+m)(l-m+1)) \delta{ellp ell} \delta{mp (m-1)}

        The spinsfast function ladder_operator_coefficient gives
        ladder_operator_coefficient(l,m) = np.sqrt((l-m)(l+m+1))
        """

        sign = float(sign)

        if (sign != 1.) and (sign != -1.):
            raise ValueError("sign must be either 1. or -1. in make_j_plus_minus")

        def j_pm_M(ell_min, ell_max):
            for ell in range(ell_min, ell_max+1):
                for m in range(-ell, ell+1):
                    mp = round(m + 1 * sign)
                    if ((mp < -ell) or (mp > ell)):
                        continue
                    yield ell, mp, ell, m, (1.j * ladder(ell, m*sign))

        return j_pm_M

    j_plus_M  = _make_j_plus_minus(1.)
    j_minus_M = _make_j_plus_minus(-1.)

    ##############################
    # Now we can use `matrix_expectation_value` to evaluate the
    # angular momentum flux. First some plumbing.

    from .waveform_modes import WaveformModes
    from . import h as htype
    from . import hdot as hdottype
    if not isinstance(h, WaveformModes):
        raise ValueError("Momentum flux can only be calculated from a `WaveformModes` object; "
                         +"`h` is of type `{0}`.".format(type(h)))
    if (hdot is not None) and (not isinstance(hdot, WaveformModes)):
        raise ValueError("Momentum flux can only be calculated from a `WaveformModes` object; "
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

    ##############################
    # Finally do the calculation

    jdot = np.zeros((hdot.n_times, 3), dtype=float)

    _, j_plus_dot  = matrix_expectation_value( hdot, j_plus_M,  h )
    _, j_minus_dot = matrix_expectation_value( hdot, j_minus_M, h )
    _, j_z_dot     = matrix_expectation_value( hdot, j_z_M,     h )

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
