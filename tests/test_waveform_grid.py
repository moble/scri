# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import copy
import math
import numpy as np
import spherical_functions as sf
import scri.sample_waveforms as samples
import spinsfast
import pytest

# from conftest import Rs

slow = pytest.mark.slow


def test_time_translation():
    """Test pure time translation, in both time_translation and supertranslation forms"""
    dt = 1.469
    alpha00 = math.sqrt(4 * math.pi) * dt
    w1 = samples.constant_waveform()
    w2 = w1.transform(time_translation=dt)
    w3 = w1.transform(supertranslation=[alpha00])
    assert np.allclose(w1.t, w2.t + dt, rtol=0.0, atol=2e-15)
    assert np.allclose(w1.data, w2.data, rtol=0.0, atol=4e-14)
    assert np.allclose(w2.t, w3.t, rtol=0.0, atol=0.0)
    assert np.allclose(w2.data, w3.data, rtol=0.0, atol=0.0)


def test_BMS_rotation(Rs):
    """Compare full BMS transformation machinery to simple Wigner-D rotation"""
    w1 = samples.constant_waveform()
    for i, R in enumerate(Rs):
        w2 = w1.copy()
        w3 = w1.copy()
        w2.rotate_decomposition_basis(R)
        w3 = w3.transform(frame_rotation=R.components)
        assert np.allclose(w2.data, w3.data, rtol=1e-15, atol=4e-13)


@slow
def test_space_translation():
    """Compare code-transformed waveform to analytically transformed waveform"""
    print("")
    ell_max = 8
    for s in range(-2, 2 + 1):
        for ell in range(abs(s), ell_max + 1):
            print("\tWorking on spin s =", s, ", ell =", ell)
            for m in range(-ell, ell + 1):
                for space_translation in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]:
                    auxiliary_waveforms = {}
                    for i in range(s + 2):
                        auxiliary_waveforms[f"psi{4-i}_modes"] = samples.single_mode_proportional_to_time(s=i - 2)
                        auxiliary_waveforms[f"psi{4-i}_modes"].data *= 0
                    w_m1 = samples.single_mode_proportional_to_time(s=s, ell=ell, m=m).transform(
                        space_translation=space_translation,
                        **auxiliary_waveforms,
                    )
                    w_m2 = samples.single_mode_proportional_to_time_supertranslated(
                        s=s, ell=ell, m=m, space_translation=np.array(space_translation)
                    )
                    i1A = np.argmin(abs(w_m1.t - (w_m1.t[0] + 2 * np.linalg.norm(space_translation))))
                    i1B = np.argmin(abs(w_m1.t - (w_m1.t[-1] - 2 * np.linalg.norm(space_translation))))
                    i2A = np.argmin(abs(w_m2.t - w_m1.t[i1A]))
                    i2B = np.argmin(abs(w_m2.t - w_m1.t[i1B]))
                    assert np.allclose(w_m1.t[i1A : i1B + 1], w_m2.t[i2A : i2B + 1], rtol=0.0, atol=1e-16), (
                        w_m1.t[i1A],
                        w_m2.t[i2A],
                        w_m1.t[i1B],
                        w_m2.t[i2B],
                        w_m1.t[i1A : i1B + 1].shape,
                        w_m2.t[i2A : i2B + 1].shape,
                    )
                    data1 = w_m1.data[i1A : i1B + 1]
                    data2 = w_m2.data[i2A : i2B + 1]
                    assert np.allclose(data1, data2, rtol=0.0, atol=5e-14), (
                        [s, ell, m],
                        space_translation,
                        [
                            abs(data1 - data2).max(),
                            data1.ravel()[np.argmax(abs(data1 - data2))],
                            data2.ravel()[np.argmax(abs(data1 - data2))],
                        ],
                        [
                            np.unravel_index(np.argmax(abs(data1 - data2)), data1.shape)[0],
                            list(
                                sf.LM_range(abs(s), ell_max)[
                                    np.unravel_index(np.argmax(abs(data1 - data2)), data1.shape)[1]
                                ]
                            ),
                        ],
                    )


@slow
def test_hyper_translation():
    """Compare code-transformed waveform to analytically transformed waveform"""
    print("")
    ell_max = 4
    for s in range(-2, 2 + 1):
        for ell in range(abs(s), ell_max + 1):
            for m in range(-ell, ell + 1):
                print("\tWorking on spin s =", s, ", ell =", ell, ", m =", m)
                for ellpp, mpp in sf.LM_range(2, ell_max):
                    supertranslation = np.zeros((sf.LM_total_size(0, ell_max),), dtype=complex)
                    if mpp == 0:
                        supertranslation[sf.LM_index(ellpp, mpp, 0)] = 1.0
                    elif mpp < 0:
                        supertranslation[sf.LM_index(ellpp, mpp, 0)] = 1.0
                        supertranslation[sf.LM_index(ellpp, -mpp, 0)] = (-1.0) ** mpp
                    elif mpp > 0:
                        supertranslation[sf.LM_index(ellpp, mpp, 0)] = 1.0j
                        supertranslation[sf.LM_index(ellpp, -mpp, 0)] = (-1.0) ** mpp * -1.0j
                    max_displacement = abs(
                        spinsfast.salm2map(supertranslation, 0, ell_max, 4 * ell_max + 1, 4 * ell_max + 1)
                    ).max()
                    auxiliary_waveforms = {}
                    for i in range(s + 2):
                        auxiliary_waveforms[f"psi{4-i}_modes"] = samples.single_mode_proportional_to_time(s=i - 2)
                        auxiliary_waveforms[f"psi{4-i}_modes"].data *= 0
                    w_m1 = samples.single_mode_proportional_to_time(s=s, ell=ell, m=m).transform(
                        supertranslation=supertranslation,
                        **auxiliary_waveforms,
                    )
                    w_m2 = samples.single_mode_proportional_to_time_supertranslated(
                        s=s, ell=ell, m=m, supertranslation=supertranslation
                    )
                    i1A = np.argmin(abs(w_m1.t - (w_m1.t[0] + 2 * max_displacement)))
                    i1B = np.argmin(abs(w_m1.t - (w_m1.t[-1] - 2 * max_displacement)))
                    i2A = np.argmin(abs(w_m2.t - w_m1.t[i1A]))
                    i2B = np.argmin(abs(w_m2.t - w_m1.t[i1B]))
                    assert np.allclose(w_m1.t[i1A : i1B + 1], w_m2.t[i2A : i2B + 1], rtol=0.0, atol=1e-16), (
                        w_m1.t[i1A],
                        w_m2.t[i2A],
                        w_m1.t[i1B],
                        w_m2.t[i2B],
                        w_m1.t[i1A : i1B + 1].shape,
                        w_m2.t[i2A : i2B + 1].shape,
                    )
                    data1 = w_m1.data[i1A : i1B + 1]
                    data2 = w_m2.data[i2A : i2B + 1]
                    assert np.allclose(data1, data2, rtol=0.0, atol=5e-14), (
                        [s, ell, m],
                        supertranslation,
                        [
                            abs(data1 - data2).max(),
                            data1.ravel()[np.argmax(abs(data1 - data2))],
                            data2.ravel()[np.argmax(abs(data1 - data2))],
                        ],
                        [
                            np.unravel_index(np.argmax(abs(data1 - data2)), data1.shape)[0],
                            list(
                                sf.LM_range(abs(s), ell_max)[
                                    np.unravel_index(np.argmax(abs(data1 - data2)), data1.shape)[1]
                                ]
                            ),
                        ],
                    )


def test_supertranslation_inverses():
    np.random.seed(1234)
    w1 = samples.random_waveform_proportional_to_time(rotating=False)
    ell_max = 4
    for ellpp, mpp in sf.LM_range(0, ell_max):
        supertranslation = np.zeros((sf.LM_total_size(0, ell_max),), dtype=complex)
        if mpp == 0:
            supertranslation[sf.LM_index(ellpp, mpp, 0)] = 1.0
        elif mpp < 0:
            supertranslation[sf.LM_index(ellpp, mpp, 0)] = 1.0
            supertranslation[sf.LM_index(ellpp, -mpp, 0)] = (-1.0) ** mpp
        elif mpp > 0:
            supertranslation[sf.LM_index(ellpp, mpp, 0)] = 1.0j
            supertranslation[sf.LM_index(ellpp, -mpp, 0)] = (-1.0) ** mpp * -1.0j
        max_displacement = abs(spinsfast.salm2map(supertranslation, 0, ell_max, 4 * ell_max + 1, 4 * ell_max + 1)).max()
        w2 = copy.deepcopy(w1)
        w2 = w2.transform(supertranslation=supertranslation)
        w2 = w2.transform(supertranslation=-supertranslation)

        # Interpolate the original onto the new time set, because we've lost some time steps
        w1 = w1.interpolate(w2.t)

        # Check that everything agrees
        # print(abs(w1.data-w2.data).max())
        assert w1._allclose(w2, rtol=5e-10, atol=5e-14)


def test_boost_inverses():
    """Boost a waveform, invert that boost, and check that the result is identical"""
    # We'll try an easier case with boost 1e-2, then a harder case with boost 1e-1.  For the latter, we need to
    # increase the ell_max substantially.
    for beta, ell_max in [(1e-2, 8), (1e-1, 14)]:
        for v in [np.array([0.0, 0.0, beta]), np.array([0.0, beta, 0.0]), np.array([beta, 0.0, 0.0])]:
            # Construct a simple sample waveform.  This fills up all the modes with reasonable values, for a good test.
            space_translation = np.array([0.1, 0.0, 0.0])
            w1 = samples.single_mode_constant_rotation(s=-2, ell=2, m=2, omega=0.3, t_0=-10.0, t_1=10.0, dt=1.0 / 200.0)
            w1 = w1.transform(space_translation=space_translation)

            # Don't include mass scaling, because that's more subtle
            w1.m_is_scaled_out = False

            # Make a copy, transform, and transform back
            w2 = w1.copy()
            w2 = w2.transform(
                boost_velocity=v, n_theta=2 * (ell_max + 1) + 1, n_phi=2 * (ell_max + 1) + 1, ell_max=ell_max
            )
            w2 = w2.transform(boost_velocity=-v, ell_max=w1.ell_max)

            # Interpolate the original onto the new time set, because we've lost some time steps
            w1 = w1.interpolate(w2.t)

            # Check that everything agrees
            # print(abs(w1.data-w2.data).max())
            assert w1._allclose(w2, atol=1e-12, rtol=0)
