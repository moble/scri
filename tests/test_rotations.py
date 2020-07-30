# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import pytest
import numpy as np
from numpy import *
import quaternion
import spherical_functions as sf
import scri

from conftest import linear_waveform, constant_waveform, random_waveform, delta_waveform


@pytest.mark.parametrize("w", [linear_waveform, constant_waveform, random_waveform])
def test_identity_rotation(w):
    # Rotation by 1 should be identity operation
    W_in = w()
    W_out = w()
    assert W_in.ensure_validity(alter=False)
    assert W_out.ensure_validity(alter=False)
    W_out.rotate_decomposition_basis(quaternion.one)
    assert W_out.ensure_validity(alter=False)
    assert np.array_equal(W_out.t, W_in.t)
    assert np.array_equal(W_out.frame, W_in.frame)
    assert np.array_equal(W_out.data, W_in.data)
    assert np.array_equal(W_out.LM, W_in.LM)
    assert W_out.ell_min == W_in.ell_min
    assert W_out.ell_max == W_in.ell_max
    for h_in, h_out in zip(W_in.history, W_out.history[:-1]):
        assert h_in == h_out.replace(
            f"{type(W_out).__name__}_{str(W_out.num)}", f"{type(W_in).__name__}_{str(W_in.num)}"
        ) or (h_in.startswith("# ") and h_out.startswith("# "))
    assert W_out.frameType == W_in.frameType
    assert W_out.dataType == W_in.dataType
    assert W_out.r_is_scaled_out == W_in.r_is_scaled_out
    assert W_out.m_is_scaled_out == W_in.m_is_scaled_out
    assert isinstance(W_out.num, int)
    assert W_out.num != W_in.num


@pytest.mark.parametrize("w", [linear_waveform, constant_waveform, random_waveform])
def test_rotation_invariants(w):
    # A random rotation should leave everything but data and frame the
    # same (except num, of course)
    W_in = w()
    W_out = w()
    np.random.seed(hash("test_rotation_invariants") % 4294967294)  # Use mod to get in an acceptable range
    W_out.rotate_decomposition_basis(np.quaternion(*np.random.uniform(-1, 1, 4)).normalized())
    assert W_in.ensure_validity(alter=False)
    assert W_out.ensure_validity(alter=False)
    assert np.array_equal(W_out.t, W_in.t)
    assert not np.array_equal(W_out.frame, W_in.frame)  # This SHOULD change
    assert not np.array_equal(W_out.data, W_in.data)  # This SHOULD change
    assert W_out.ell_min == W_in.ell_min
    assert W_out.ell_max == W_in.ell_max
    assert np.array_equal(W_out.LM, W_in.LM)
    for h_in, h_out in zip(W_in.history[:-3], W_out.history[:-5]):
        assert h_in == h_out.replace(
            f"{type(W_out).__name__}_{str(W_out.num)}", f"{type(W_in).__name__}_{str(W_in.num)}"
        ) or (h_in.startswith("# ") and h_out.startswith("# "))
    assert W_out.frameType == W_in.frameType
    assert W_out.dataType == W_in.dataType
    assert W_out.r_is_scaled_out == W_in.r_is_scaled_out
    assert W_out.m_is_scaled_out == W_in.m_is_scaled_out
    assert W_out.num != W_in.num


@pytest.mark.parametrize("w", [linear_waveform, constant_waveform, random_waveform])
def test_constant_versus_series(w):
    # A random rotation should leave everything but data and frame the
    # same (except num, of course)
    W_const = w()
    W_series = w()
    np.random.seed(hash("test_constant_versus_series") % 4294967294)  # Use mod to get in an acceptable range
    W_const.rotate_decomposition_basis(np.quaternion(*np.random.uniform(-1, 1, 4)).normalized())
    W_series.rotate_decomposition_basis(
        np.array([np.quaternion(*np.random.uniform(-1, 1, 4)).normalized()] * W_series.n_times)
    )
    assert W_const.ensure_validity(alter=False)
    assert W_series.ensure_validity(alter=False)
    assert np.array_equal(W_series.t, W_const.t)
    assert not np.array_equal(W_series.frame, W_const.frame)  # This SHOULD change
    assert not np.array_equal(W_series.data, W_const.data)  # This SHOULD change
    assert W_series.ell_min == W_const.ell_min
    assert W_series.ell_max == W_const.ell_max
    assert np.array_equal(W_series.LM, W_const.LM)
    for h_const, h_series in zip(W_const.history[:-5], W_series.history[:-11]):
        assert h_const == h_series.replace(
            f"{type(W_series).__name__}_{str(W_series.num)}", f"{type(W_const).__name__}_{str(W_const.num)}"
        ) or (h_const.startswith("# ") and h_series.startswith("# "))
    assert W_series.frameType == W_const.frameType
    assert W_series.dataType == W_const.dataType
    assert W_series.r_is_scaled_out == W_const.r_is_scaled_out
    assert W_series.m_is_scaled_out == W_const.m_is_scaled_out
    assert W_series.num != W_const.num


@pytest.mark.parametrize("w", [linear_waveform, constant_waveform, random_waveform])
def test_rotation_inversion(w):
    # Rotation followed by the inverse rotation should leave
    # everything the same (except that the frame data will be either a
    # 1 or a series of 1s)
    np.random.seed(hash("test_rotation_inversion") % 4294967294)  # Use mod to get in an acceptable range
    W_in = w()
    assert W_in.ensure_validity(alter=False)
    # We loop over (1) a single constant rotation, and (2) an array of random rotations
    for R_basis in [
        np.quaternion(*np.random.uniform(-1, 1, 4)).normalized(),
        np.array([np.quaternion(*np.random.uniform(-1, 1, 4)).normalized()] * W_in.n_times),
    ]:
        W_out = w()
        W_out.rotate_decomposition_basis(R_basis)
        W_out.rotate_decomposition_basis(~R_basis)
        assert W_out.ensure_validity(alter=False)
        assert np.array_equal(W_out.t, W_in.t)
        assert np.max(np.abs(W_out.frame - W_in.frame)) < 1e-15
        assert np.allclose(W_out.data, W_in.data, atol=W_in.ell_max ** 4 ** 4e-14, rtol=W_in.ell_max ** 4 * 4e-14)
        assert W_out.ell_min == W_in.ell_min
        assert W_out.ell_max == W_in.ell_max
        assert np.array_equal(W_out.LM, W_in.LM)
        for h_in, h_out in zip(W_in.history[:-3], W_out.history[:-5]):
            assert h_in == h_out.replace(
                f"{type(W_out).__name__}_{str(W_out.num)}", f"{type(W_in).__name__}_{str(W_in.num)}"
            ) or (h_in.startswith("# datetime") and h_out.startswith("# datetime"))
        assert W_out.frameType == W_in.frameType
        assert W_out.dataType == W_in.dataType
        assert W_out.r_is_scaled_out == W_in.r_is_scaled_out
        assert W_out.m_is_scaled_out == W_in.m_is_scaled_out
        assert W_out.num != W_in.num


def test_rotations_of_0_0_mode(Rs):
    # The (ell,m)=(0,0) mode should be rotationally invariant
    n_copies = 10
    W_in = delta_waveform(0, 0, begin=-10.0, end=100.0, n_times=n_copies * len(Rs), ell_min=0, ell_max=8)
    assert W_in.ensure_validity(alter=False)
    W_out = scri.WaveformModes(W_in)
    R_basis = np.array([R for R in Rs for i in range(n_copies)])
    W_out.rotate_decomposition_basis(R_basis)
    assert W_out.ensure_validity(alter=False)
    assert np.array_equal(W_out.t, W_in.t)
    assert np.max(np.abs(W_out.frame - R_basis)) == 0.0
    assert np.array_equal(W_out.data, W_in.data)
    assert W_out.ell_min == W_in.ell_min
    assert W_out.ell_max == W_in.ell_max
    assert np.array_equal(W_out.LM, W_in.LM)
    for h_in, h_out in zip(W_in.history, W_out.history[:-1]):
        assert h_in == h_out.replace(
            f"{type(W_out).__name__}_{str(W_out.num)}", f"{type(W_in).__name__}_{str(W_in.num)}"
        ) or (h_in.startswith("# ") and h_out.startswith("# "))
    assert W_out.frameType == W_in.frameType
    assert W_out.dataType == W_in.dataType
    assert W_out.r_is_scaled_out == W_in.r_is_scaled_out
    assert W_out.m_is_scaled_out == W_in.m_is_scaled_out
    assert W_out.num != W_in.num


def test_rotations_of_each_mode_individually(Rs):
    ell_min = 0
    ell_max = 8  # sf.ell_max is just too much; this test is too slow, and ell=8 should be fine
    R_basis = Rs
    Ds = np.empty((len(Rs), sf.LMpM_total_size(ell_min, ell_max)), dtype=complex)
    for i, R in enumerate(Rs):
        Ds[i, :] = sf.Wigner_D_matrices(R, ell_min, ell_max)
    for ell in range(ell_max + 1):
        first_zeros = np.zeros((len(Rs), sf.LM_total_size(ell_min, ell - 1)), dtype=complex)
        later_zeros = np.zeros((len(Rs), sf.LM_total_size(ell + 1, ell_max)), dtype=complex)
        for Mp in range(-ell, ell):
            W_in = delta_waveform(ell, Mp, begin=-10.0, end=100.0, n_times=len(Rs), ell_min=ell_min, ell_max=ell_max)
            # Now, the modes are f^{\ell,m[} = \delta^{\ell,mp}_{L,Mp}
            assert W_in.ensure_validity(alter=False)
            W_out = scri.WaveformModes(W_in)
            W_out.rotate_decomposition_basis(Rs)
            assert W_out.ensure_validity(alter=False)
            assert np.array_equal(W_out.t, W_in.t)
            assert np.max(np.abs(W_out.frame - R_basis)) == 0.0
            i_D0 = sf.LMpM_index(ell, Mp, -ell, ell_min)
            assert np.array_equal(W_out.data[:, : sf.LM_total_size(ell_min, ell - 1)], first_zeros)
            if ell < ell_max:
                assert np.array_equal(
                    W_out.data[:, sf.LM_total_size(ell_min, ell - 1) : -sf.LM_total_size(ell + 1, ell_max)],
                    Ds[:, i_D0 : i_D0 + (2 * ell + 1)],
                )
                assert np.array_equal(W_out.data[:, -sf.LM_total_size(ell + 1, ell_max) :], later_zeros)
            else:
                assert np.array_equal(
                    W_out.data[:, sf.LM_total_size(ell_min, ell - 1) :], Ds[:, i_D0 : i_D0 + (2 * ell + 1)]
                )
            assert W_out.ell_min == W_in.ell_min
            assert W_out.ell_max == W_in.ell_max
            assert np.array_equal(W_out.LM, W_in.LM)
            for h_in, h_out in zip(W_in.history, W_out.history[:-1]):
                assert h_in == h_out.replace(type(W_out).__name__ + str(W_out.num), type(W_in).__name__ + str(W_in.num))
            assert W_out.frameType == W_in.frameType
            assert W_out.dataType == W_in.dataType
            assert W_out.r_is_scaled_out == W_in.r_is_scaled_out
            assert W_out.m_is_scaled_out == W_in.m_is_scaled_out
            assert W_out.num != W_in.num
