import math
import numpy as np
import quaternion
import spinsfast
import spherical_functions as sf
import scri
import pytest

ABD = scri.AsymptoticBondiData
abd = scri.asymptotic_bondi_data


def kerr_schild(mass, spin, ell_max=8):
    psi2 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)
    psi1 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)
    psi0 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)

    psi2[0] = -sf.constant_as_ell_0_mode(mass)
    psi1[2] = (3j * spin / 2) * (np.sqrt((8 / 3) * np.pi))
    psi0[6] = (3 * spin ** 2 / mass / 2) * (np.sqrt((32 / 15) * np.pi))

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
    # print('Bondi-violation norms', abd.bondi_violation_norms)


def test_abd_conformal_factors():
    tolerance = 4e-14
    ell_max = 32
    n_theta = 2 * ell_max + 1
    n_phi = n_theta
    cf = abd.transformations.conformal_factors

    def cf2(v, distorted_grid_rotors):
        from math import sin, cos

        # κ = 1 / [γ(1-v⋅r)]
        # ð(κ⁻¹) = - κ⁻² ðκ
        # ðκ = - κ² ð(κ⁻¹)
        theta_phi = sf.theta_phi(n_theta, n_phi)
        kinv = γ * np.array(
            [
                [1 - v[0] * sin(θ) * cos(ϕ) - v[1] * math.sin(θ) * math.sin(ϕ) - v[2] * math.cos(θ) for θ, ϕ in row]
                for row in theta_phi
            ]
        )
        k = 1 / kinv
        κinv = spinsfast.map2salm(kinv, 0, ell_max)
        κ = spinsfast.map2salm(k, 0, ell_max)
        SWSHs = sf.SWSH_grid(distorted_grid_rotors, 0, ell_max)
        one_over_k2 = np.tensordot(κinv, SWSHs, axes=([-1], [-1]))
        one_over_k_cubed2 = one_over_k2 ** 3
        k2 = 1 / one_over_k2
        ðκ = sf.eth_GHP(κ, 0)
        SWSHs = sf.SWSH_grid(distorted_grid_rotors, 1, ell_max)
        ðk = np.tensordot(ðκ, SWSHs, axes=([-1], [-1]))
        ðk_over_k2 = ðk / k2
        return (
            k2[np.newaxis, :, :],
            ðk_over_k2[np.newaxis, :, :],
            one_over_k2[np.newaxis, :, :],
            one_over_k_cubed2[np.newaxis, :, :],
        )

    frame_rotation = quaternion.one
    boost_velocity = np.array([0.01, 0.02, 0.03])
    β = np.linalg.norm(boost_velocity)
    γ = 1 / math.sqrt(1 - β ** 2)
    distorted_grid_rotors = abd.transformations.boosted_grid(frame_rotation, boost_velocity, n_theta, n_phi)
    k, ðk_over_k, one_over_k, one_over_k_cubed = cf(boost_velocity, distorted_grid_rotors)
    k2, ðk_over_k2, one_over_k2, one_over_k_cubed2 = cf2(boost_velocity, distorted_grid_rotors)
    assert k.shape == k2.shape
    assert ðk_over_k.shape == ðk_over_k2.shape
    assert one_over_k.shape == one_over_k2.shape
    assert one_over_k_cubed.shape == one_over_k_cubed2.shape
    assert isinstance(k, sf.Grid)
    assert isinstance(ðk_over_k, sf.Grid)
    assert isinstance(one_over_k, sf.Grid)
    assert isinstance(one_over_k_cubed, sf.Grid)
    assert k.s == 0
    assert ðk_over_k.s == 1
    assert one_over_k.s == 0
    assert one_over_k_cubed.s == 0
    assert np.allclose(k.view(np.ndarray), k2, atol=tolerance, rtol=tolerance)
    assert np.allclose(ðk_over_k.view(np.ndarray), ðk_over_k2, atol=tolerance, rtol=tolerance)
    assert np.allclose(one_over_k.view(np.ndarray), one_over_k2, atol=tolerance, rtol=tolerance)
    assert np.allclose(one_over_k_cubed.view(np.ndarray), one_over_k_cubed2, atol=tolerance, rtol=tolerance)


def test_abd_schwarzschild_transform():
    tolerance = 1e-14
    for v in [np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.1, 0.0]), np.array([0.0, 0.0, 0.1])]:
        mass = 1.0
        ell_max = 8
        u = np.linspace(0, 100, num=5000)
        psi2, psi1, psi0 = kerr_schild(mass, 0.0, ell_max)
        abd = ABD.from_initial_values(u, ell_max=ell_max, psi2=psi2)
        β = np.linalg.norm(v)
        γ = 1 / np.sqrt(1 - β ** 2)
        abdprime = abd.transform(boost_velocity=v)
        transformed_four_momentum = abdprime.bondi_four_momentum
        expected_four_momentum = mass * γ * np.array([1,] + v.tolist())
        # print()
        # print(f"v={v}, β={β}, γ={γ}, βγ={β*γ}")
        # print(f"Expected four-momentum:\n{expected_four_momentum}")
        # print(f"New four-momentum:\n{transformed_four_momentum}")
        assert np.allclose(expected_four_momentum, transformed_four_momentum, atol=tolerance, rtol=tolerance)
