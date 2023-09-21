# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import math
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
    assert (
        max(
            np.amin(
                np.vstack((np.linalg.norm(LL - LL_expected, axis=1), np.linalg.norm(LL + LL_expected, axis=1))), axis=0
            )
        )
        < 1.0e-14
    )


@pytest.mark.parametrize("w", [linear_waveform, constant_waveform, random_waveform])
def test_dpa_rotated_generally(w, Rs):
    from scri.mode_calculations import LLDominantEigenvector

    n_copies = 10
    W = w(begin=1.0, end=100.0, n_times=n_copies * len(Rs), ell_min=0, ell_max=8)
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
    assert (
        max(np.amin(np.vstack((np.linalg.norm(LL1 - LL2, axis=1), np.linalg.norm(LL1 + LL2, axis=1))), axis=0))
        < 1.0e-12
    )


def test_zero_angular_velocity():
    from scri.mode_calculations import angular_velocity

    w = constant_waveform(end=10.0, n_times=10000)
    Omega_out = angular_velocity(w)
    assert np.allclose(Omega_out, np.zeros_like(Omega_out), atol=1e-15, rtol=0.0)


def test_z_angular_velocity():
    from scri.mode_calculations import angular_velocity

    w = constant_waveform(end=10.0, n_times=10000)
    omega = 2 * math.pi / 5.0
    R = np.exp(quaternion.quaternion(0, 0, 0, omega / 2) * w.t)
    w.rotate_physical_system(R)
    Omega_out = angular_velocity(w)
    Omega_in = np.zeros_like(Omega_out)
    Omega_in[:, 2] = omega
    assert np.allclose(Omega_in, Omega_out, atol=1e-12, rtol=2e-8)


def test_rotated_angular_velocity():
    from scri.mode_calculations import angular_velocity

    w = constant_waveform(end=10.0, n_times=10000)
    omega = 2 * math.pi / 5.0
    R0 = quaternion.quaternion(1, 2, 3, 4).normalized()
    R = R0 * np.exp(quaternion.quaternion(0, 0, 0, omega / 2) * w.t)
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
    omega = 2 * math.pi / 5.0
    R0 = quaternion.quaternion(1, 2, 3, 4).normalized()
    R_in = R0 * np.exp(quaternion.quaternion(0, 0, 0, omega / 2) * w.t)
    w_rot = w.deepcopy()
    w_rot.rotate_physical_system(R_in)
    R_out = corotating_frame(w_rot, R0=R0, tolerance=1e-12)
    assert np.allclose(quaternion.as_float_array(R_in), quaternion.as_float_array(R_out), atol=1e-10, rtol=0.0)
    w_rot.to_corotating_frame(R0=R0, tolerance=1e-12)
    assert w._allclose(w_rot, atol=1e-8)
    assert w_rot.frameType == Corotating
