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


def test_abd_kerr_superrest_frame():
    mass = 2.0
    spin = 0.456
    ell_max = 8
    u = np.linspace(-100, 100, num=5000)
    psi2, psi1, psi0 = kerr_schild(mass, spin, ell_max)
    abd = ABD.from_initial_values(u, ell_max=ell_max, psi2=psi2, psi1=psi1)

    tolerance = 1e-12

    supertranslation = np.array(
        [0.0, 3e-2 - 1j * 5e-3, 1e-3, -3e-2 - 1j * 5e-3, 2e-4 + 1j * 1e-4, 1j * 3e-3, 1e-2, 1j * 3e-3, 2e-4 - 1j * 1e-4]
    )
    frame_rotation = quaternion.quaternion(1, 2, 3, 4).normalized().components
    boost_velocity = np.array([2e-4, -3e-5, 2e-4])
    abd_prime = abd.transform(
        supertranslation=supertranslation, frame_rotation=frame_rotation, boost_velocity=boost_velocity
    )

    abd_recovered, transformation, rel_errs = abd_prime.map_to_superrest_frame(
        t_0=0,
        padding_time=20,
        N_itr_maxes={"superrest": 1, "CoM_transformation": 10, "rotation": 10, "supertranslation": 10},
    )

    PsiM_recovered = abd_recovered.supermomentum("Moreschi")[np.argmin(abs(abd_recovered.t))]
    PsiM_recovered[0:4] = 0
    PsiM_recovered_S2 = spinsfast.salm2map(
        PsiM_recovered.view(np.ndarray), 0, ell_max, 2 * ell_max + 1, 2 * ell_max + 1
    )
    PsiM_recovered_norm = spinsfast.map2salm((abs(PsiM_recovered_S2) ** 2).view(np.ndarray), 0, ell_max)[0] / np.sqrt(
        4 * np.pi
    )
    assert np.allclose(PsiM_recovered_norm, 0.0, atol=tolerance, rtol=tolerance)

    chi_recovered = abd_recovered.bondi_dimensionless_spin()
    chi_recovered = chi_recovered / np.linalg.norm(chi_recovered, axis=-1)[:, None]
    assert np.allclose(chi_recovered, [[0, 0, 1]] * abd_recovered.t.size, atol=tolerance, rtol=tolerance)

    G_recovered = abd_recovered.bondi_CoM_charge() / abd_recovered.bondi_four_momentum()[:, 0, None]
    assert np.allclose(G_recovered[np.argmin(abs(abd_recovered.t))], 0.0, atol=tolerance, rtol=tolerance)
    assert np.allclose(
        derivative(G_recovered, abd_recovered.t)[np.argmin(abs(abd_recovered.t))], 0.0, atol=tolerance, rtol=tolerance
    )


def test_abd_kerr_target_superrest_frame():
    mass = 2.0
    spin = 0.456
    ell_max = 8
    u = np.linspace(-100, 100, num=5000)
    psi2, psi1, psi0 = kerr_schild(mass, spin, ell_max)
    abd = ABD.from_initial_values(u, ell_max=ell_max, psi2=psi2, psi1=psi1)

    tolerance = 1e-12

    supertranslation = np.array(
        [0.0, 3e-2 - 1j * 5e-3, 1e-3, -3e-2 - 1j * 5e-3, 2e-4 + 1j * 1e-4, 1j * 3e-3, 1e-2, 1j * 3e-3, 2e-4 - 1j * 1e-4]
    )
    abd_target = abd.transform(
        supertranslation=supertranslation,
    )

    abd_recovered, transformation, rel_errs = abd.map_to_superrest_frame(
        t_0=0,
        padding_time=20,
        target_PsiM_input=abd_target.supermomentum("Moreschi"),
        N_itr_maxes={
            "superrest": 1,
            "CoM_transformation": 10,
            "rotation": 10,
            "supertranslation": 10,
        },
    )

    PsiM_diff = abd_recovered.supermomentum("Moreschi")[np.argmin(abs(abd_recovered.t))]
    PsiM_diff -= abd_target.supermomentum("Moreschi")[np.argmin(abs(abd_recovered.t))]
    PsiM_diff[0:4] = 0
    PsiM_diff_S2 = spinsfast.salm2map(PsiM_diff.view(np.ndarray), 0, ell_max, 2 * ell_max + 1, 2 * ell_max + 1)
    PsiM_diff_norm = spinsfast.map2salm((abs(PsiM_diff_S2) ** 2).view(np.ndarray), 0, ell_max)[0] / np.sqrt(4 * np.pi)
    assert np.allclose(PsiM_diff_norm, 0.0, atol=tolerance, rtol=tolerance)

    chi_recovered = abd_recovered.bondi_dimensionless_spin()
    chi_recovered = chi_recovered / np.linalg.norm(chi_recovered, axis=-1)[:, None]
    assert np.allclose(chi_recovered, [[0, 0, 1]] * abd_recovered.t.size, atol=tolerance, rtol=tolerance)

    G = abd_recovered.bondi_CoM_charge() / abd_recovered.bondi_four_momentum()[:, 0, None]
    assert np.allclose(G[np.argmin(abs(abd_recovered.t))], 0.0, atol=tolerance, rtol=tolerance)
    assert np.allclose(
        derivative(G, abd_recovered.t)[np.argmin(abs(abd_recovered.t))], 0.0, atol=tolerance, rtol=tolerance
    )
