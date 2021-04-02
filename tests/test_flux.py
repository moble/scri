# Copyright (c) 2019, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import numpy as np
import quaternion
import spinsfast
import scri
from numpy import *
import pytest

from conftest import linear_waveform, constant_waveform, random_waveform


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
    hdot = h.data_dot
    zeros = np.zeros((hdot.shape[0], 4), dtype=hdot.dtype)
    data = np.concatenate((zeros, hdot), axis=1)  # Pad with zeros for spinsfast
    ell_min = 0
    ell_max = 2 * h.ell_max + 1  # Maximum ell value required for nhat*|hdot|^2
    n_theta = 2 * ell_max + 1
    n_phi = n_theta
    hdot_map = spinsfast.salm2map(data, h.spin_weight, h.ell_max, n_theta, n_phi)
    hdot_mag_squared_map = hdot_map * hdot_map.conjugate()

    theta = np.linspace(0.0, np.pi, num=n_theta, endpoint=True)
    phi = np.linspace(0.0, 2 * np.pi, num=n_phi, endpoint=False)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    pdot = np.array(
        [
            spinsfast.map2salm(hdot_mag_squared_map * n / (16 * np.pi), 0, 0)[..., 0].real * (2 * np.sqrt(np.pi))
            for n in [x, y, z]
        ]
    ).T

    return pdot


def silly_angular_momentum_flux(h, hdot=None):
    """Compute angular momentum flux from waveform explicitly

    This function uses very explicit (and slow) methods, but different as far as possible from the
    methods used in the main code.

    """
    hdot = hdot or h.data_dot
    zeros = np.zeros((hdot.shape[0], 4), dtype=hdot.dtype)
    data = np.concatenate((zeros, hdot), axis=1)  # Pad with zeros for spinsfast
    ell_min = 0
    ell_max = 2 * h.ell_max  # Maximum ell value required to fully resolve Ji{h} * hdot
    n_theta = 2 * ell_max + 1
    n_phi = n_theta
    hdot_map = spinsfast.salm2map(data, h.spin_weight, h.ell_max, n_theta, n_phi)

    jdot = np.zeros((h.n_times, 3), dtype=float)

    # Compute J_+ h
    for ell in range(h.ell_min, h.ell_max + 1):
        i = h.index(ell, -ell) + 4
        data[:, i] = 0.0j
        for m in range(-ell, ell):
            i = h.index(ell, m + 1) + 4
            j = h.index(ell, m)
            data[:, i] = 1.0j * np.sqrt((ell - m) * (ell + m + 1)) * h.data[:, j]
    jplus_h_map = spinsfast.salm2map(data, h.spin_weight, h.ell_max, n_theta, n_phi)

    # Compute J_- h
    for ell in range(h.ell_min, h.ell_max + 1):
        for m in range(-ell + 1, ell + 1):
            i = h.index(ell, m - 1) + 4
            j = h.index(ell, m)
            data[:, i] = 1.0j * np.sqrt((ell + m) * (ell - m + 1)) * h.data[:, j]
        i = h.index(ell, ell) + 4
        data[:, i] = 0.0j
    jminus_h_map = spinsfast.salm2map(data, h.spin_weight, h.ell_max, n_theta, n_phi)

    # Combine jplus and jminus to compute x-y components, then conjugate, multiply by hdot, and integrate
    jx_h_map = 0.5 * (jplus_h_map + jminus_h_map)
    jy_h_map = -0.5j * (jplus_h_map - jminus_h_map)
    jdot[:, 0] = (
        -spinsfast.map2salm(jx_h_map.conjugate() * hdot_map, 0, 0)[..., 0].real * (2 * np.sqrt(np.pi)) / (16 * np.pi)
    )
    jdot[:, 1] = (
        -spinsfast.map2salm(jy_h_map.conjugate() * hdot_map, 0, 0)[..., 0].real * (2 * np.sqrt(np.pi)) / (16 * np.pi)
    )

    # Compute J_z h, then conjugate, multiply by hdot, and integrate
    for ell in range(h.ell_min, h.ell_max + 1):
        for m in range(-ell, ell + 1):
            i = h.index(ell, m)
            data[:, i + 4] = 1.0j * m * h.data[:, i]
    jz_h_map = spinsfast.salm2map(data, h.spin_weight, h.ell_max, n_theta, n_phi)
    jdot[:, 2] = (
        -spinsfast.map2salm(jz_h_map.conjugate() * hdot_map, 0, 0)[..., 0].real * (2 * np.sqrt(np.pi)) / (16 * np.pi)
    )

    return jdot


def test_momentum_flux():
    h = scri.sample_waveforms.fake_precessing_waveform(t_1=1_000.0)
    pdot1 = silly_momentum_flux(h)
    pdot2 = scri.momentum_flux(h)
    assert np.allclose(pdot1, pdot2, rtol=1e-13, atol=1e-13)


def test_angular_momentum_flux():
    h = scri.sample_waveforms.fake_precessing_waveform(t_1=1_000.0)
    jdot1 = silly_angular_momentum_flux(h)
    jdot2 = scri.angular_momentum_flux(h)
    assert np.allclose(jdot1, jdot2, rtol=1e-13, atol=1e-13)
    

def test_boost_flux():
    from quaternion import rotate_vectors    

    h = scri.sample_waveforms.single_mode(ell=8, m=5)  # Any mode can be used to implement this
    R = np.quaternion(1, 4, 3, 2).normalized()
    A = h.boost_flux()
    for i in range(len(A)):
        A[i, :] = rotate_vectors(R, A[i,:], axis=-1)
    B = (h.rotate_decomposition_basis(~R)).boost_flux()
    assert np.allclose(A, B, rtol=1e-13, atol=1e-13)
