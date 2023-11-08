import scri
import pytest
import numpy as np
import spinsfast
import spherical_functions as sf
import quaternion
from quaternion.calculus import derivative

ABD = scri.AsymptoticBondiData


def kerr_schild(mass, spin, ell_max=8):
    psi2 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)
    psi1 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)
    psi0 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)

    # In the Moreschi-Boyle convention
    psi2[0] = -sf.constant_as_ell_0_mode(mass)
    psi1[2] = -np.sqrt(2) * (3j * spin / 2) * (np.sqrt((8 / 3) * np.pi))
    psi0[6] = 2 * (3 * spin**2 / mass / 2) * (np.sqrt((32 / 15) * np.pi))

    return psi2, psi1, psi0


def test_abd_to_abd():
    mass = 2.0
    spin = 0.456
    ell_max = 8
    u = np.linspace(-100, 100, num=5000)
    psi2, psi1, psi0 = kerr_schild(mass, spin, ell_max)
    abd = ABD.from_initial_values(u, ell_max=ell_max, psi2=psi2, psi1=psi1)

    supertranslation = np.array(
        [0.0, 3e-2 - 1j * 5e-3, 1e-3, -3e-2 - 1j * 5e-3, 2e-4 + 1j * 1e-4, 1j * 3e-3, 1e-2, 1j * 3e-3, 2e-4 - 1j * 1e-4]
    )
    frame_rotation = quaternion.quaternion(1, 2, 3, 4).normalized().components
    boost_velocity = np.array([2e-4, -3e-4, -5e-4])

    abd_target = abd.transform(
        supertranslation=supertranslation, frame_rotation=frame_rotation, boost_velocity=boost_velocity
    )

    # We don't perform time/phase fixing because this data is radiation-free so it doesn't make sense
    abd_recovered, transformation, rel_err = abd.map_to_abd_frame(
        abd_target,
        t_0=0,
        padding_time=20,
        N_itr_maxes={"abd": 2, "superrest": 1, "CoM_transformation": 10, "rotation": 10, "supertranslation": 10},
        fix_time_phase_freedom=False,
        nprocs=-1,
    )

    abd_target_interp = abd_target.interpolate(
        abd_target.t[
            np.argmin(abs(abd_target.t - max(abd_target.t[0], abd_recovered.t[0]))) : np.argmin(
                abs(abd_target.t - min(abd_target.t[-1], abd_recovered.t[-1]))
            )
            + 1
        ]
    )
    abd_recovered_interp = abd_recovered.interpolate(
        abd_recovered.t[
            np.argmin(abs(abd_recovered.t - max(abd_target.t[0], abd_recovered.t[0]))) : np.argmin(
                abs(abd_recovered.t - min(abd_target.t[-1], abd_recovered.t[-1]))
            )
            + 1
        ]
    )

    assert np.allclose(
        np.array(
            [
                abd_target_interp.sigma,
                abd_target_interp.psi4,
                abd_target_interp.psi3,
                abd_target_interp.psi2,
                abd_target_interp.psi1,
                abd_target_interp.psi0,
            ]
        ),
        np.array(
            [
                abd_recovered_interp.sigma,
                abd_recovered_interp.psi4,
                abd_recovered_interp.psi3,
                abd_recovered_interp.psi2,
                abd_recovered_interp.psi1,
                abd_recovered_interp.psi0,
            ]
        ),
    )
