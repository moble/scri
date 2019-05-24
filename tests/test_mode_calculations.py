# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

import numpy as np
import quaternion
import spinsfast
from numpy import *
import pytest

from conftest import linear_waveform, constant_waveform, random_waveform


@pytest.mark.parametrize("w", [linear_waveform, constant_waveform])
def test_dpa_simple_cases(w):
    from scri.mode_calculations import LLDominantEigenvector

    LL = LLDominantEigenvector(w())
    LL_expected = np.zeros_like(LL, dtype=float)
    LL_expected[:, 2] = 1.0
    assert np.allclose(LL, LL_expected)


@pytest.mark.parametrize("w", [linear_waveform, constant_waveform])
def test_dpa_rotated_simple_cases(w, Rs):
    from scri.mode_calculations import LLDominantEigenvector
    # We use `begin=1.0` because we need to avoid situations where the modes
    # are all zeros, which can happen in `linear_waveform` at t=0.0
    W = w(begin=1.0, ell_min=0, n_times=len(Rs))
    W.rotate_physical_system(Rs)
    LL = LLDominantEigenvector(W)
    LL_expected = quaternion.as_float_array(Rs * np.array([quaternion.z for i in range(len(Rs))]) * (~Rs))[:, 1:]

    # Because the dpa is only defined up to a sign, all we need is for the
    # dot product between the dpa and the expected value to be close to
    # either 1 or -1.  This finds the largest difference, based on the
    # smaller of the two sign options.
    assert max(
        np.amin(
            np.vstack(
                (np.linalg.norm(LL - LL_expected, axis=1),
                 np.linalg.norm(LL + LL_expected, axis=1))),
            axis=0)
    ) < 1.e-14


@pytest.mark.parametrize("w", [linear_waveform, constant_waveform, random_waveform])
def test_dpa_rotated_generally(w, Rs):
    from scri.mode_calculations import LLDominantEigenvector

    n_copies = 10
    W = w(begin=1., end=100., n_times=n_copies * len(Rs), ell_min=0, ell_max=8)
    assert W.ensure_validity(alter=False)
    R_basis = np.array([R for R in Rs for i in range(n_copies)])

    # We use `begin=1.0` because we need to avoid situations where the modes
    # are all zeros, which can happen in `linear_waveform` at t=0.0
    LL1 = LLDominantEigenvector(W)
    LL1 = quaternion.as_float_array(np.array([R * np.quaternion(0, *v) * (~R) for R, v in zip(R_basis, LL1)]))[:, 1:]
    W.rotate_physical_system(R_basis)
    LL2 = LLDominantEigenvector(W)

    # Because the dpa is only defined up to a sign, all we need is for the
    # dot product between the dpa and the expected value to be close to
    # either 1 or -1.  This finds the largest difference, based on the
    # smaller of the two sign options.
    assert max(
        np.amin(
            np.vstack(
                (np.linalg.norm(LL1 - LL2, axis=1),
                 np.linalg.norm(LL1 + LL2, axis=1))),
            axis=0)
    ) < 1.e-12


def test_zero_angular_velocity():
    from scri.mode_calculations import angular_velocity

    w = constant_waveform(end=10.0, n_times=10000)
    Omega_out = angular_velocity(w)
    assert np.allclose(Omega_out, np.zeros_like(Omega_out), atol=1e-15, rtol=0.0)


def test_z_angular_velocity():
    from scri.mode_calculations import angular_velocity

    w = constant_waveform(end=10.0, n_times=10000)
    omega = 2*math.pi/5.0
    R = np.exp(quaternion.quaternion(0, 0, 0, omega/2)*w.t)
    w.rotate_physical_system(R)
    Omega_out = angular_velocity(w)
    Omega_in = np.zeros_like(Omega_out)
    Omega_in[:, 2] = omega
    assert np.allclose(Omega_in, Omega_out, atol=1e-12, rtol=2e-8)


def test_rotated_angular_velocity():
    from scri.mode_calculations import angular_velocity

    w = constant_waveform(end=10.0, n_times=10000)
    omega = 2*math.pi/5.0
    R0 = quaternion.quaternion(1, 2, 3, 4).normalized()
    R = R0 * np.exp(quaternion.quaternion(0, 0, 0, omega/2)*w.t)
    w.rotate_physical_system(R)
    Omega = R0 * quaternion.quaternion(0, 0, 0, omega) * R0.inverse()
    Omega_out = angular_velocity(w)
    Omega_in = np.zeros_like(Omega_out)
    Omega_in[:, 0] = Omega.x
    Omega_in[:, 1] = Omega.y
    Omega_in[:, 2] = Omega.z
    assert np.allclose(Omega_in, Omega_out, atol=1e-12, rtol=2e-8)


def test_corotating_frame():
    from scri.mode_calculations import corotating_frame
    from scri import Corotating

    w = constant_waveform(end=10.0, n_times=100000)  # Need lots of time steps for accurate integration
    omega = 2*math.pi/5.0
    R0 = quaternion.quaternion(1, 2, 3, 4).normalized()
    R_in = R0 * np.exp(quaternion.quaternion(0, 0, 0, omega/2)*w.t)
    w_rot = w.deepcopy()
    w_rot.rotate_physical_system(R_in)
    R_out = corotating_frame(w_rot, R0=R0, tolerance=1e-12)
    assert np.allclose(quaternion.as_float_array(R_in), quaternion.as_float_array(R_out), atol=1e-10, rtol=0.0)
    w_rot.to_corotating_frame(R0=R0, tolerance=1e-12)
    assert w._allclose(w_rot, atol=1e-8)
    assert w_rot.frameType == Corotating


def silly_momentum_flux(h):
    """Compute momentum flux from waveform with a silly but simple method

    This function evaluates the momentum-flux formula quite literally.  The formula is

         dp       R**2  | dh |**2 ^
    ---------- = ------ | -- |    n
    dOmega  dt   16  pi | dt |

    Here, p and nhat are vectors and R is the distance to the source.  The input h is differentiated
    numerically to find modes of dh/dt, which are then used to construct the values of dh/dt on a
    grid.  At each point of that grid, the value of |dh/dt|**2 nhat is computed, which is then
    integrated over the sphere to arrive at dp/dt.  Note that this integration is accomplished by
    way of a spherical-harmonic decomposition; the ell=m=0 mode is multiplied by 2*sqrt(pi) to
    arrive at the result that would be achieved by integrating over the sphere.

    """
    import spinsfast
    hdot = h.data_dot
    zeros = np.zeros((hdot.shape[0], 4), dtype=hdot.dtype)
    data = np.concatenate((zeros, hdot), axis=1)  # Pad with zeros for spinsfast
    ell_min = 0
    ell_max = 2*h.ell_max + 1  # Maximum ell value required for nhat*|hdot|^2
    n_theta = 2*ell_max + 1
    n_phi = n_theta
    hdot_map = spinsfast.salm2map(data, h.spin_weight, h.ell_max, n_theta, n_phi)
    hdot_mag_squared_map = hdot_map * hdot_map.conjugate()

    theta = np.linspace(0.0, np.pi, num=n_theta, endpoint=True)
    phi = np.linspace(0.0, 2*np.pi, num=n_phi, endpoint=False)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    pdot = np.array([spinsfast.map2salm(hdot_mag_squared_map*n/(16*np.pi), 0, 0)[..., 0] * (2*np.sqrt(np.pi))
                     for n in [x, y, z]]).T

    return pdot


def test_momentum_flux():
    import numpy as np
    import scri
    directory = '/Users/boyle/Research/Data/SimulationAnnex/CatalogLinks/SXS:BBH:0030/Lev5/'
    h = scri.SpEC.read_from_h5(directory+'rhOverM_Asymptotic_GeometricUnits_CoM.h5/Extrapolated_N2.dir')
    pdot1 = silly_momentum_flux(h)
    pdot2 = scri.momentum_flux(h)

    # diff = np.linalg.norm(pdot1-pdot2, axis=1)
    # ind = np.argmax(diff, axis=None)  # np.unravel_index(np.argmax(diff, axis=None), diff.shape)
    # print('Max diff norm:', diff[ind], pdot1[ind], pdot2[ind], ind)

    assert np.allclose(pdot1, pdot2, rtol=1e-13, atol=1e-13)
