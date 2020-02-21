import math
import numpy as np
import quaternion
import spinsfast
import spherical_functions as sf
import scri
import pytest

ABD = scri.AsymptoticBondiData


def kerr_schild(mass, spin, ell_max=8):
    psi2 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)
    psi1 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)
    psi0 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)

    psi2[0] = -sf.constant_as_ell_0_mode(mass)
    psi1[2] = (3j * spin / 2) * (np.sqrt((8/3) * np.pi))
    psi0[6] = (3 * spin**2 / mass / 2) * (np.sqrt((32/15) * np.pi))

    return psi2, psi1, psi0


def test_abd_schwarzschild():
    mass = 0.789
    ell_max = 8
    u = np.linspace(0, 100, num=5000)
    psi2, psi1, psi0 = kerr_schild(mass, 0.0, ell_max)
    abd = ABD.from_initial_values(u, ell_max=ell_max, psi2=psi2)
    expected_four_momentum = np.array([mass, 0, 0, 0])
    computed_four_momentum = abd.bondi_four_momentum
    # print()
    # print(f"Expected four-momentum: {expected_four_momentum}")
    # print(f"Computed four-momentum: {computed_four_momentum}")
    assert np.allclose(computed_four_momentum, expected_four_momentum, atol=1e-14, rtol=1e-14)
    #print('Bondi-violation norms', abd.bondi_violation_norms)


def test_abd_schwarzschild_transform():
    tolerance = 1e-14
    for v in [np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.1, 0.0]), np.array([0.0, 0.0, 0.1])]:
        mass = 1.0
        ell_max = 8
        u = np.linspace(0, 100, num=5000)
        psi2, psi1, psi0 = kerr_schild(mass, 0.0, ell_max)
        abd = ABD.from_initial_values(u, ell_max=ell_max, psi2=psi2)
        β = np.linalg.norm(v)
        γ = 1 / np.sqrt(1 - β**2)
        abdprime = abd.transform(boost_velocity=v)
        transformed_four_momentum = abdprime.bondi_four_momentum
        expected_four_momentum = mass * γ * np.array([1,] + v.tolist())
        # print()
        # print(f"v={v}, β={β}, γ={γ}, βγ={β*γ}")
        # print(f"Expected four-momentum:\n{expected_four_momentum}")
        # print(f"New four-momentum:\n{transformed_four_momentum}")
        assert np.allclose(expected_four_momentum, transformed_four_momentum, atol=tolerance, rtol=tolerance)
