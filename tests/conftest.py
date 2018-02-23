# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

import pytest

import numpy as np
from numpy import *
import quaternion
import spherical_functions as sf

import scri


def pytest_addoption(parser):
    parser.addoption("--run_slow_tests", action="store_true",
                     help="Run all tests, including slow ones")


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getoption("--run_slow_tests"):
        pytest.skip("Need `--run_slow_tests` command-line argument to run")


@pytest.fixture
def constant_waveform(begin=-10., end=100., n_times=1000, ell_min=2, ell_max=8):
    t = np.linspace(begin, end, num=n_times)
    frame = np.array([quaternion.x for t_i in t])
    lm = np.array([[ell, m] for ell in range(ell_min, ell_max + 1) for m in range(-ell, ell + 1)])
    data = np.empty((t.shape[0], lm.shape[0]), dtype=complex)
    for i, m in enumerate(lm[:, 1]):
        data[:, i] = (m - 1j * m)
    W = scri.WaveformModes(t=t, frame=frame, data=data,
                           ell_min=min(lm[:, 0]), ell_max=max(lm[:, 0]),
                           history=['# Called from constant_waveform'],
                           frameType=scri.Corotating, dataType=scri.h,
                           r_is_scaled_out=True, m_is_scaled_out=True)
    return W


@pytest.fixture
def linear_waveform(begin=-10., end=100., n_times=1000, ell_min=2, ell_max=8):
    np.random.seed(hash('linear_waveform') % 4294967294)  # Use mod to get in an acceptable range
    axis = np.quaternion(0., *np.random.uniform(-1, 1, size=3)).normalized()
    t = np.linspace(begin, end, num=n_times)
    omega = 2 * np.pi * 4 / (t[-1] - t[0])
    frame = np.array([np.exp(axis * (omega * t_i / 2)) for t_i in t])
    lm = np.array([[ell, m] for ell in range(ell_min, ell_max + 1) for m in range(-ell, ell + 1)])
    data = np.empty((t.shape[0], lm.shape[0]), dtype=complex)
    for i, m in enumerate(lm[:, 1]):
        # N.B.: This form is used in test_linear_interpolation; if you
        # change it here, you must change it there.
        data[:, i] = (m - 1j * m) * t
    W = scri.WaveformModes(t=t, frame=frame, data=data,
                           ell_min=min(lm[:, 0]), ell_max=max(lm[:, 0]),
                           history=['# Called from linear_waveform'],
                           frameType=scri.Corotating, dataType=scri.h,
                           r_is_scaled_out=True, m_is_scaled_out=True)
    return W


@pytest.fixture
def random_waveform(begin=-10., end=100., n_times=1000, ell_min=None, ell_max=8, dataType=scri.h):
    np.random.seed(hash('random_waveform') % 4294967294)  # Use mod to get in an acceptable range
    spin_weight = scri.SpinWeights[scri.DataType.index(dataType)]
    if ell_min is None:
        ell_min = abs(spin_weight)
    n_modes = (ell_max * (ell_max + 2) - ell_min ** 2 + 1)
    t = np.sort(np.random.uniform(begin, end, size=n_times))
    frame = np.array([np.quaternion(*np.random.uniform(-1, 1, 4)).normalized() for t_i in t])
    data = np.random.normal(size=(n_times, n_modes, 2)).view(complex)[:, :, 0]
    W = scri.WaveformModes(t=t, frame=frame, data=data,
                           ell_min=ell_min, ell_max=ell_max,
                           history=['# Called from random_waveform'],
                           frameType=scri.Corotating, dataType=dataType,
                           r_is_scaled_out=True, m_is_scaled_out=False)
    return W


@pytest.fixture
def delta_waveform(ell, m, begin=-10., end=100., n_times=1000, ell_min=2, ell_max=8):
    """WaveformModes with 1 in selected slot and 0 elsewhere"""
    n_modes = (ell_max * (ell_max + 2) - ell_min ** 2 + 1)
    t = np.linspace(begin, end, num=n_times)
    data = np.zeros((n_times, n_modes), dtype=complex)
    data[:, sf.LM_index(ell, m, ell_min)] = 1.0 + 0.0j
    W = scri.WaveformModes(t=t, data=data,  # frame=frame,
                           ell_min=ell_min, ell_max=ell_max,
                           history=['# Called from delta_waveform'],
                           frameType=scri.Inertial, dataType=scri.psi4,
                           r_is_scaled_out=False, m_is_scaled_out=True)
    return W


@pytest.fixture
def special_angles():
    return np.arange(-1 * np.pi, 1 * np.pi + 0.1, np.pi / 4.)


@pytest.fixture
def Rs():
    ones = [0, -1., 1.]
    rs = [np.quaternion(w, x, y, z).normalized() for w in ones for x in ones for y in ones for z in ones][1:]
    np.random.seed(hash('Rs') % 4294967294)  # Use mod to get in an acceptable range
    rs = rs + [np.quaternion(*np.random.uniform(-1, 1, 4)).normalized() for i in range(20)]
    return np.array(rs)

