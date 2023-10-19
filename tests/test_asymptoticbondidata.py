import math
import numpy as np
import quaternion
import spinsfast
import spherical_functions as sf
import scri
import pytest

from conftest import kerr_schild, real_supertranslation

ABD = scri.AsymptoticBondiData
abd = scri.asymptotic_bondi_data


def test_abd_schwarzschild():
    mass = 0.789
    ell_max = 8
    u = np.linspace(0, 100, num=5000)
    psi2, psi1, psi0 = kerr_schild(mass, 0.0, ell_max)
    abd = ABD.from_initial_values(u, ell_max=ell_max, psi2=psi2)
    expected_four_momentum = np.array([mass, 0, 0, 0])
    computed_four_momentum = abd.bondi_four_momentum()
    expected_angular_momentum = np.zeros(3)
    computed_angular_momentum = abd.bondi_angular_momentum()
    # print()
    # print(f"Expected four-momentum: {expected_four_momentum}")
    # print(f"Computed four-momentum: {computed_four_momentum}")
    assert np.allclose(computed_four_momentum, expected_four_momentum, atol=1e-14, rtol=1e-14)
    assert np.allclose(computed_angular_momentum, expected_angular_momentum, atol=1e-14, rtol=1e-14)
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
        one_over_k_cubed2 = one_over_k2**3
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
    γ = 1 / math.sqrt(1 - β**2)
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
        rest_mass = abd.bondi_rest_mass()
        β = np.linalg.norm(v)
        γ = 1 / np.sqrt(1 - β**2)
        abdprime = abd.transform(boost_velocity=v)
        transformed_rest_mass = abdprime.bondi_rest_mass()
        transformed_four_momentum = abdprime.bondi_four_momentum()
        expected_four_momentum = mass * γ * np.array([1, *-v])
        # print()
        # print(f"v={v}, β={β}, γ={γ}, βγ={β*γ}")
        # print(f"Expected four-momentum:\n{expected_four_momentum}")
        # print(f"New four-momentum:\n{transformed_four_momentum}")
        assert np.allclose(expected_four_momentum, transformed_four_momentum, atol=tolerance, rtol=tolerance)
        assert np.allclose(rest_mass[0], transformed_rest_mass, atol=tolerance, rtol=tolerance)


def test_abd_bondi_angular_momentum():
    tolerance = 1e-14
    for v in [
        np.array([0.1, 0.0, 0.0]),
        np.array([0.0, 0.1, 0.0]),
        np.array([0.1, 0.1, 0.1]),
        np.array([0.0, 0.0, 0.1]),
    ]:
        mass = 1.0
        spin = 0.456
        ell_max = 8
        u = np.linspace(0, 100, num=5000)
        psi2, psi1, psi0 = kerr_schild(mass, spin, ell_max)
        abd = ABD.from_initial_values(u, ell_max=ell_max, psi2=psi2, psi1=psi1)
        β = np.linalg.norm(v)
        γ = 1 / np.sqrt(1 - β**2)
        abdprime = abd.transform(boost_velocity=v)
        transformed_angular_momentum = abdprime.bondi_angular_momentum()
        angular_momentum = abd.bondi_angular_momentum()[0]
        expected_angular_momentum = γ * angular_momentum + (1 - γ) * np.dot(angular_momentum, v / β) * (v / β)
        assert np.allclose(expected_angular_momentum, transformed_angular_momentum, atol=tolerance, rtol=tolerance)


def test_abd_kerr():
    v = np.array([0.085, -0.034, 0.1])
    mass = 2.0
    spin = 0.456
    ell_max = 8
    u = np.linspace(0, 100, num=5000)
    psi2, psi1, psi0 = kerr_schild(mass, spin, ell_max)
    abd = ABD.from_initial_values(u, ell_max=ell_max, psi2=psi2, psi1=psi1)
    angular_momentum = abd.bondi_angular_momentum()
    S = abd.bondi_dimensionless_spin()
    tolerance = 1e-14
    # This is true because we are in the center-of-momentum frame
    assert np.allclose(S * mass**2, angular_momentum, atol=tolerance, rtol=tolerance)
    abdprime = abd.transform(boost_velocity=v)
    S_prime = abdprime.bondi_dimensionless_spin()
    tolerance = 1e-14
    assert np.allclose(S_prime[-1], S[-1], atol=tolerance, rtol=tolerance)
    N = abdprime.bondi_boost_charge()
    G = abdprime.bondi_CoM_charge()
    P = abdprime.bondi_four_momentum()
    assert np.allclose(N, G - abdprime.t[:, np.newaxis] * P[:, 1:], atol=tolerance, rtol=tolerance)


def test_abd_WaveformModes():
    # Compare a general BMS transformation of a ~random ABD object to the equivalent transformation
    # of the strain of a WaveformModes object

    tolerance = 4e-12 # should pass with 9e-13; bumped for safety; would be 1e-14 if not for boost
    np.random.seed(123)

    def abd_to_h_data(abd):
        # h = 2 σ̄
        return 2 * abd.sigma.bar.ndarray.copy()

    # Generate initial ABD from Kerr-Schild plus some quadratic-in-time shear data
    mass = 1.23
    spin = 0.35
    ell_max = 6
    u = np.linspace(-10, 10, num=1_000)
    ψ2, ψ1, ψ0 = kerr_schild(mass, spin, ell_max)
    σ0 = np.squeeze(0.01000 * (np.random.rand((ell_max+1)**2, 2).view(complex) - (0.5+0.5j)))
    σ0[:sf.LM_index(2,-2,0)] = 0
    σ̇0 = np.squeeze(0.00020 * (np.random.rand((ell_max+1)**2, 2).view(complex) - (0.5+0.5j)))
    σ̇0[:sf.LM_index(2,-2,0)] = 0
    σ̈0 = np.squeeze(0.00003 * (np.random.rand((ell_max+1)**2, 2).view(complex) - (0.5+0.5j)))
    σ̈0[:sf.LM_index(2,-2,0)] = 0
    abd = ABD.from_initial_values(u, ell_max=ell_max, sigma0=σ0, sigmadot0=σ̇0, sigmaddot0=σ̈0, psi2=ψ2, psi1=ψ1, psi0=ψ0)
    h = scri.WaveformModes(
        t=abd.t, data=abd_to_h_data(abd), ell_min=abd.ell_min, ell_max=abd.ell_max,
        frameType=scri.Inertial, dataType=scri.h, r_is_scaled_out=True, m_is_scaled_out=True
    )

    # Generate a random supertranslation
    α = real_supertranslation(np.squeeze(0.01 * (np.random.rand((ell_max+1)**2, 2).view(complex) - (0.5+0.5j))))
    R = np.random.randn(4)  # Will be normalized automatically
    v⃗ = 0.01 * (np.random.rand(3)-0.5)

    # Transform ABD and comparable WaveformModes objects
    abdprime = abd.transform(supertranslation=α, frame_rotation=R, boost_velocity=v⃗)
    hprime = h.transform(supertranslation=α, frame_rotation=R, boost_velocity=v⃗)

    # Interpolate to same set of times
    t = scri.extrapolation.intersection(hprime.t, abdprime.t)
    hprime = hprime.interpolate(t)
    abdprime = abdprime.interpolate(t)

    assert np.allclose(
        hprime.data,
        abd_to_h_data(abdprime)[:, 4:],
        atol=tolerance, rtol=tolerance
    )
