# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import numpy as np
import quaternion
from numpy import *
import pytest
import scri
from scri.waveform_modes import WaveformModes

from conftest import linear_waveform, constant_waveform, random_waveform


def test_empty_WaveformModes():
    W = scri.WaveformModes()
    assert W.ensure_validity(alter=False)
    assert W.t.shape == (0,)
    assert W.frame.shape == (0,)
    assert W.data.shape == (0, 0)
    assert W.LM.shape == (0, 2)
    assert W.ell_min == 0
    assert W.ell_max == -1
    # assert W.history == ['WaveformModes22 = WaveformModes(...)']
    assert W.frameType == scri.UnknownFrameType
    assert W.dataType == scri.UnknownDataType
    assert not W.r_is_scaled_out  # != True
    assert not W.m_is_scaled_out  # != True


@pytest.mark.xfail
def test_bad_construction():
    # both with and without override_exception_from_invalidity
    assert False


@pytest.mark.xfail
def test_unused_constructor_kwarg():
    assert False


def test_nontrivial_construction(linear_waveform):
    assert linear_waveform is not None  # As long as this was constructed, we're okay


def test_pickling():
    import pickle

    W1 = scri.WaveformModes()
    W1_str = pickle.dumps(W1)
    W2 = pickle.loads(W1_str)
    assert "_WaveformBase__num" in W1.__dict__.keys()
    assert "_WaveformBase__num" in W2.__dict__.keys()
    assert W1._allclose(W2, rtol=0, atol=0, compare_history_beginnings=True)


def test_waveform_properties(linear_waveform):
    W = linear_waveform
    assert W.ensure_validity(alter=False)
    assert W.n_data_sets == len([1 for ell in range(2, 9) for m in range(-ell, ell + 1)])
    assert W.n_times == 1000
    assert W.spin_weight == -2
    assert W.conformal_weight == -1
    assert W.gamma_weight == -1
    assert W.r_scaling == 1
    assert W.m_scaling == 0
    assert W.frame_type_string == "Corotating"
    assert W.data_type_string == "h"
    assert W.data_type_latex == "h"
    assert W.descriptor_string == "rhOverM"
    assert W.ell_min == 2
    assert W.ell_max == 8
    assert W.ells == (2, 8)
    assert np.array_equal(W.LM, np.array([[ell, m] for ell in range(2, 9) for m in range(-ell, ell + 1)]))


def test_indexing(linear_waveform):
    w = linear_waveform
    for i, (ell, m) in enumerate(w.LM):
        assert w.index(ell, m) == i
        assert np.allclose(w.data[:, w.index(ell, m)], (m - 1j * m) * w.t, rtol=0, atol=0)
    assert np.all(w.indices(w.LM) == range(w.n_modes))


def test_empty_slice(linear_waveform):
    W = linear_waveform[:0, :0]
    assert W.ensure_validity(alter=False)
    # Check that these things are copied from linear_waveform
    assert W.history[:1] == ["# Called from linear_waveform"]
    assert W.frameType == scri.Corotating
    assert W.dataType == scri.h
    assert W.r_is_scaled_out  # == True
    assert W.m_is_scaled_out  # == True
    assert W.num != linear_waveform.num
    # Check that these are as in an empty WaveformModes
    assert W.t.shape == (0,)
    assert W.frame.shape == (0,)
    assert W.data.shape == (0, 0)
    assert W.ell_min == 0
    assert W.ell_max == -1


def test_empty_mode_slice(linear_waveform):
    W = linear_waveform[:, :0]
    assert W.ensure_validity(alter=False)
    # Check that these things are copied from linear_waveform
    assert np.all(W.t == linear_waveform.t)
    assert np.all(W.frame == linear_waveform.frame)
    # Check that these are empty
    assert W.data.size == 0
    assert W.LM.size == 0
    assert W.ell_min == 0
    assert W.ell_max == -1
    assert W.history[:1] == ["# Called from linear_waveform"]
    assert W.frameType == linear_waveform.frameType
    assert W.dataType == linear_waveform.dataType
    assert W.r_is_scaled_out == linear_waveform.r_is_scaled_out
    assert W.m_is_scaled_out == linear_waveform.m_is_scaled_out
    assert W.num != linear_waveform.num  # Check that this is valid, but not the same


def test_time_slice(linear_waveform):
    W = linear_waveform[10:50]
    assert W.ensure_validity(alter=False)
    assert np.all(W.t == linear_waveform.t[10:50])
    assert np.all(W.frame == linear_waveform.frame[10:50])
    assert np.all(W.data == linear_waveform.data[10:50])
    assert np.all(W.LM == linear_waveform.LM)
    assert W.ell_min == linear_waveform.ell_min
    assert W.ell_max == linear_waveform.ell_max
    assert W.history[:1] == ["# Called from linear_waveform"]
    assert W.frameType == linear_waveform.frameType
    assert W.dataType == linear_waveform.dataType
    assert W.r_is_scaled_out == linear_waveform.r_is_scaled_out
    assert W.m_is_scaled_out == linear_waveform.m_is_scaled_out
    assert isinstance(W.num, int)
    assert W.num != linear_waveform.num


def test_time_and_mode_slice(linear_waveform):
    W = linear_waveform[10:50, :5]
    assert W.ensure_validity(alter=False)
    assert W.history[:1] == ["# Called from linear_waveform"]
    assert W.frameType == scri.Corotating
    assert W.dataType == scri.h
    assert W.r_is_scaled_out  # == True
    assert W.m_is_scaled_out  # == True
    assert W.num != linear_waveform.num
    assert np.all(W.t == linear_waveform.t[10:50])
    assert np.all(W.frame == linear_waveform.frame[10:50])
    assert np.all(
        W.LM == np.array([[ell, m] for ell in range(linear_waveform.ell_min, 5) for m in range(-ell, ell + 1)])
    )
    assert np.all(W.data == linear_waveform.data[10:50, :21])


def test_norms(linear_waveform):
    W = linear_waveform
    W.data[0, :] = 6.0 * W.data[-1, :]
    W.data[10, :] = 5.0 * W.data[-1, :]
    assert W.ensure_validity(alter=False)
    assert np.allclose(W.norm(), np.sum(np.abs(W.data) ** 2, axis=-1), rtol=1.0e-15)
    assert np.allclose(W.norm(take_sqrt=True), np.sqrt(np.sum(np.abs(W.data) ** 2, axis=-1)), rtol=1.0e-15)
    assert np.allclose(
        W.norm(indices=slice(W.data.shape[0] // 4, None, None)),
        np.sum(np.abs(W.data[(W.data.shape[0] // 4) :]) ** 2, axis=-1),
        rtol=1.0e-15,
    )
    assert np.allclose(
        W.norm(indices=slice(W.data.shape[0] // 4, None, None), take_sqrt=True),
        np.sqrt(np.sum(np.abs(W.data[(W.data.shape[0] // 4) :]) ** 2, axis=-1)),
        rtol=1.0e-15,
    )
    assert W.max_norm_index() == W.data.shape[0] - 1
    assert W.max_norm_index(0) == 0
    assert W.max_norm_index(1) == 0
    assert W.max_norm_index(W.data.shape[0]) == 10
    assert W.max_norm_time() == W.t[-1]
    assert W.max_norm_time(0) == W.t[0]
    assert W.max_norm_time(1) == W.t[0]
    assert W.max_norm_time(W.data.shape[0]) == W.t[10]


@pytest.mark.parametrize("W", [linear_waveform, constant_waveform])
def test_interpolation(W):
    interpolation_precision = 4.5e-16

    # Interpolation onto self.t is the identity (except for `num`)
    W_in = W()
    t_in = W_in.t
    t_out = np.copy(t_in)
    W_out = W_in.interpolate(t_out)
    assert W_in.ensure_validity(alter=False)
    assert W_out.ensure_validity(alter=False)
    assert W_out.history[:2] == W_in.history[:2]
    assert W_out.frameType == W_in.frameType
    assert W_out.dataType == W_in.dataType
    assert W_out.r_is_scaled_out == W_in.r_is_scaled_out
    assert W_out.m_is_scaled_out == W_in.m_is_scaled_out
    assert W_out.num != W_in.num
    assert np.all(W_out.t == W_in.t)
    assert np.all(W_out.frame == W_in.frame)
    assert np.all(W_out.LM == W_in.LM)
    assert np.allclose(W_out.data, W_in.data, rtol=4.5e-16)

    # Interpolation onto another time series should reshape
    W_in = W()
    t_in = W_in.t
    t_out = (t_in[:-1] + t_in[1:]) / 2.0
    W_out = W_in.interpolate(t_out)
    assert W_in.ensure_validity(alter=False)
    assert W_out.ensure_validity(alter=False)
    assert W_out.history[:1] == [f"# Called from {W.__name__}"]
    assert W_out.frameType == scri.Corotating
    assert W_out.dataType == scri.h
    assert W_out.r_is_scaled_out  # == True
    assert W_out.m_is_scaled_out  # == True
    assert isinstance(W_out.num, int)
    assert W_out.num != W_in.num
    assert np.all(W_out.t == t_out)
    assert np.all(W_out.frame == quaternion.squad(W_in.frame, W_in.t, W_out.t))
    assert np.all(W_out.LM == W_in.LM)
    assert W_out.data.shape == (len(t_out), W_in.n_data_sets)


def test_constant_interpolation(constant_waveform):
    interpolation_precision = 4.5e-16
    # Interpolation of constant_waveform is ~identity
    W_in = constant_waveform
    t_in = W_in.t
    t_out = (t_in[:-1] + t_in[1:]) / 2.0
    W_out = W_in.interpolate(t_out)
    assert W_in.ensure_validity(alter=False)
    assert W_out.ensure_validity(alter=False)
    assert W_out.history[:1] == ["# Called from constant_waveform"]
    assert W_out.frameType == scri.Corotating
    assert W_out.dataType == scri.h
    assert W_out.r_is_scaled_out  # == True
    assert W_out.m_is_scaled_out  # == True
    assert W_out.num != W_in.num
    assert np.all(W_out.t == t_out)
    assert np.all(W_out.frame == np.array([W_in.frame[0],] * len(t_out)))
    assert np.all(W_out.LM == W_in.LM)
    assert np.allclose(W_out.data, np.array([W_in.data[0, :],] * len(t_out)), rtol=interpolation_precision)


def test_linear_interpolation(linear_waveform):
    interpolation_precision = 4.5e-16
    # Interpolation of constant_waveform is ~identity
    W_in = linear_waveform
    t_in = W_in.t
    t_out = (t_in[:-1] + t_in[1:]) / 2.0
    W_out = W_in.interpolate(t_out)
    assert W_in.ensure_validity(alter=False)
    assert W_out.ensure_validity(alter=False)
    assert W_out.history[:1] == ["# Called from linear_waveform"]
    assert W_out.frameType == scri.Corotating
    assert W_out.dataType == scri.h
    assert W_out.r_is_scaled_out  # == True
    assert W_out.m_is_scaled_out  # == True
    assert W_out.num != W_in.num
    assert np.all(W_out.t == t_out)
    squad = quaternion.squad(W_in.frame, W_in.t, t_out)
    assert np.all(np.abs(W_out.frame - squad) < interpolation_precision * np.abs(W_out.frame + squad))
    assert np.all(W_out.LM == W_in.LM)
    data = np.empty((t_out.shape[0], W_in.LM.shape[0]), dtype=complex)
    for i, m in enumerate(W_in.LM[:, 1]):
        # N.B.: This depends on the construction of linear_waveform;
        # only change here if it is changed there.
        data[:, i] = (m - 1j * m) * t_out
    assert np.allclose(W_out.data, data, rtol=interpolation_precision)


def test_involution_properties(random_waveform):
    """Ensure that involutions truly are involute

    The objective here is to apply the involutions twice, and ensure that we get the same thing out.

    """
    for direction in ["x_", "y_", "z_", ""]:
        w_in = WaveformModes(random_waveform)
        w_out = getattr(getattr(w_in, f"{direction}parity_conjugate"), f"{direction}parity_conjugate")
        assert w_in._allclose(w_out, rtol=0.0, atol=0.0, compare_history_beginnings=True)


def test_idempotents(random_waveform):
    """Ensure that (anti)symmetrization composed with itself is itself

    Taking the symmetric part of the symmetric part should be the same as just the original symmetric part.  The same
    goes for anti-symmetric parts.

    """
    for direction in ["x_", "y_", "z_", ""]:
        w_in = getattr(random_waveform, f"{direction}parity_symmetric_part")
        w_out = getattr(w_in, f"{direction}parity_symmetric_part")
        assert w_in._allclose(w_out, rtol=0.0, atol=0.0, compare_history_beginnings=True)


def test_null_compositions(random_waveform):
    """Ensure that symmetric parts of antisymmetric parts are zero, and vice versa

    Both the `frame` and `data` members should be zeroed out by such operations.

    """
    w_in = WaveformModes(random_waveform)
    w_in.data = np.zeros_like(w_in.data)
    w_in.frame = np.zeros_like(w_in.frame)
    for direction in ["x_", "y_", "z_", ""]:
        w_out = getattr(
            getattr(random_waveform, f"{direction}parity_symmetric_part"), f"{direction}parity_antisymmetric_part"
        )
        assert w_in._allclose(w_out, rtol=0.0, atol=0.0)
    for direction in ["x_", "y_", "z_", ""]:
        w_out = getattr(
            getattr(random_waveform, f"{direction}parity_antisymmetric_part"), f"{direction}parity_symmetric_part"
        )
        assert w_in._allclose(w_out, rtol=0.0, atol=0.0)


def test_parity_violation_measures(random_waveform):
    """Apply parity-part operators, then measure parity violation

    The parity-violation of the parity-symmetric part should be zero.  The parity-violation of the original waveform
    should equal the norm of the parity-antisymmetric part (or 1 for the normalized violation).

    """
    w_in = WaveformModes(random_waveform)
    zeros = np.zeros_like(w_in.data[:, 0])
    ones = np.ones_like(w_in.data[:, 0])
    for direction in ["x_", "y_", "z_", ""]:
        w_out = getattr(random_waveform, f"{direction}parity_symmetric_part")
        assert np.allclose(getattr(w_out, f"{direction}parity_violation_squared"), zeros, atol=1e-15)
        assert np.allclose(getattr(w_out, f"{direction}parity_violation_normalized"), zeros, atol=1e-15)
    for direction in ["x_", "y_", "z_", ""]:
        w_out = getattr(random_waveform, f"{direction}parity_antisymmetric_part")
        assert np.allclose(getattr(w_in, f"{direction}parity_violation_squared"), w_out.norm(), atol=0.0, rtol=1e-15)
        assert np.allclose(
            getattr(w_in, f"{direction}parity_violation_normalized"),
            np.sqrt(w_out.norm() / w_in.norm()),
            atol=0.0,
            rtol=1e-15,
        )
        assert np.allclose(getattr(w_out, f"{direction}parity_violation_normalized"), ones, atol=0.0, rtol=1e-15)


@pytest.mark.xfail
def test_involutions():
    assert False


def test_SI_units(linear_waveform):
    units_precision = 4.5e-16
    TotalMassInSolarMasses = 1.1
    DistanceInMegaparsecs = 3.7
    SpeedOfLight = 299792458  # Units of meters / seconds
    SolarMassInSeconds = 4.92549094916e-06  # Units of seconds
    MassInSeconds = TotalMassInSolarMasses * SolarMassInSeconds
    OneMegaparsec = 3.0856775814913672789e22  # Units of meters
    DistanceInMeters = DistanceInMegaparsecs * OneMegaparsec
    W_in = linear_waveform
    W_out = W_in.SI_units(TotalMassInSolarMasses, DistanceInMegaparsecs)
    assert W_in.ensure_validity(alter=False)
    assert W_out.ensure_validity(alter=False)
    assert W_out.history[:1] == ["# Called from linear_waveform"]
    assert W_out.frameType == scri.Corotating
    assert W_out.dataType == scri.h
    assert not W_out.r_is_scaled_out  # != True
    assert not W_out.m_is_scaled_out  # != True
    assert W_out.num != W_in.num
    assert np.allclose(W_out.t, W_in.t * MassInSeconds, rtol=units_precision)
    assert np.allclose(W_out.data, W_in.data * MassInSeconds * SpeedOfLight / DistanceInMeters, rtol=units_precision)
