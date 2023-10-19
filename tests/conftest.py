# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import pytest

import numpy as np
from numpy import *
import quaternion
import spherical_functions as sf

import scri


def pytest_addoption(parser):
    parser.addoption("--run_slow_tests", action="store_true", help="Run all tests, including slow ones")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_slow_tests"):
        # --run_slow_tests given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --run_slow_tests option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getoption("--run_slow_tests"):
        pytest.skip("Need `--run_slow_tests` command-line argument to run")


def kerr_schild(mass, spin, ell_max=8):
    psi2 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)
    psi1 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)
    psi0 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)

    # In the Moreschi-Boyle convention
    psi2[0] = -sf.constant_as_ell_0_mode(mass)
    psi1[2] = -np.sqrt(2) * (3j * spin / 2) * (np.sqrt((8 / 3) * np.pi))
    psi0[6] = 2 * (3 * spin**2 / mass / 2) * (np.sqrt((32 / 15) * np.pi))

    return psi2, psi1, psi0


@pytest.fixture(name="kerr_schild")
def kerr_schild_fixture():
    return kerr_schild()


def constant_waveform(begin=-10.0, end=100.0, n_times=1000, ell_min=2, ell_max=8):
    t = np.linspace(begin, end, num=n_times)
    frame = np.array([quaternion.x for t_i in t])
    lm = np.array([[ell, m] for ell in range(ell_min, ell_max + 1) for m in range(-ell, ell + 1)])
    data = np.empty((t.shape[0], lm.shape[0]), dtype=complex)
    for i, m in enumerate(lm[:, 1]):
        data[:, i] = m - 1j * m
    W = scri.WaveformModes(
        t=t,
        frame=frame,
        data=data,
        ell_min=min(lm[:, 0]),
        ell_max=max(lm[:, 0]),
        history=["# Called from constant_waveform"],
        frameType=scri.Corotating,
        dataType=scri.h,
        r_is_scaled_out=True,
        m_is_scaled_out=True,
    )
    return W


@pytest.fixture(name="constant_waveform")
def constant_waveform_fixture():
    return constant_waveform()


def linear_waveform(begin=-10.0, end=100.0, n_times=1000, ell_min=2, ell_max=8):
    np.random.seed(hash("linear_waveform") % 4294967294)  # Use mod to get in an acceptable range
    axis = np.quaternion(0.0, *np.random.uniform(-1, 1, size=3)).normalized()
    t = np.linspace(begin, end, num=n_times)
    omega = 2 * np.pi * 4 / (t[-1] - t[0])
    frame = np.array([np.exp(axis * (omega * t_i / 2)) for t_i in t])
    lm = np.array([[ell, m] for ell in range(ell_min, ell_max + 1) for m in range(-ell, ell + 1)])
    data = np.empty((t.shape[0], lm.shape[0]), dtype=complex)
    for i, m in enumerate(lm[:, 1]):
        # N.B.: This form is used in test_linear_interpolation; if you
        # change it here, you must change it there.
        data[:, i] = (m - 1j * m) * t
    W = scri.WaveformModes(
        t=t,
        frame=frame,
        data=data,
        ell_min=min(lm[:, 0]),
        ell_max=max(lm[:, 0]),
        history=["# Called from linear_waveform"],
        frameType=scri.Corotating,
        dataType=scri.h,
        r_is_scaled_out=True,
        m_is_scaled_out=True,
    )
    return W


@pytest.fixture(name="linear_waveform")
def linear_waveform_fixture():
    return linear_waveform()


def random_waveform(begin=-10.0, end=100.0, n_times=1000, ell_min=None, ell_max=8, dataType=scri.h):
    np.random.seed(hash("random_waveform") % 4294967294)  # Use mod to get in an acceptable range
    spin_weight = scri.SpinWeights[scri.DataType.index(dataType)]
    if ell_min is None:
        ell_min = abs(spin_weight)
    n_modes = ell_max * (ell_max + 2) - ell_min**2 + 1
    t = np.sort(np.random.uniform(begin, end, size=n_times))
    frame = np.array([np.quaternion(*np.random.uniform(-1, 1, 4)).normalized() for t_i in t])
    data = np.random.normal(size=(n_times, n_modes, 2)).view(complex)[:, :, 0]
    W = scri.WaveformModes(
        t=t,
        frame=frame,
        data=data,
        ell_min=ell_min,
        ell_max=ell_max,
        history=["# Called from random_waveform"],
        frameType=scri.Corotating,
        dataType=dataType,
        r_is_scaled_out=True,
        m_is_scaled_out=False,
    )
    return W


@pytest.fixture(name="random_waveform")
def random_waveform_fixture():
    return random_waveform()


def delta_waveform(ell, m, begin=-10.0, end=100.0, n_times=1000, ell_min=2, ell_max=8):
    """WaveformModes with 1 in selected slot and 0 elsewhere"""
    n_modes = ell_max * (ell_max + 2) - ell_min**2 + 1
    t = np.linspace(begin, end, num=n_times)
    data = np.zeros((n_times, n_modes), dtype=complex)
    data[:, sf.LM_index(ell, m, ell_min)] = 1.0 + 0.0j
    W = scri.WaveformModes(
        t=t,
        data=data,  # frame=frame,
        ell_min=ell_min,
        ell_max=ell_max,
        history=["# Called from delta_waveform"],
        frameType=scri.Inertial,
        dataType=scri.psi4,
        r_is_scaled_out=False,
        m_is_scaled_out=True,
    )
    return W


@pytest.fixture(name="delta_waveform")
def delta_waveform_fixture():
    return delta_waveform()


@pytest.fixture
def special_angles():
    return np.arange(-1 * np.pi, 1 * np.pi + 0.1, np.pi / 4.0)


@pytest.fixture
def Rs():
    ones = [0, -1.0, 1.0]
    rs = [np.quaternion(w, x, y, z).normalized() for w in ones for x in ones for y in ones for z in ones][1:]
    np.random.seed(hash("Rs") % 4294967294)  # Use mod to get in an acceptable range
    rs = rs + [np.quaternion(*np.random.uniform(-1, 1, 4)).normalized() for i in range(20)]
    return np.array(rs)


def real_supertranslation(α):
    """Update α in place to represent a real-valued supertranslation."""
    ℓ = 0
    while True:
        for m in range(-ℓ, ℓ+1):
            iₚ, iₘ = sf.LM_index(ℓ, m, 0), sf.LM_index(ℓ, -m, 0)
            if iₚ >= α.size:
                return α
            α[iₚ] = (α[iₚ] + (-1.0)**m * α[iₘ].conj()) / 2
            α[iₘ] = (-1.0)**m * α[iₚ].conj()
        ℓ += 1
