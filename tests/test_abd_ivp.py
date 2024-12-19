import math
import numpy as np
import quaternion
import spinsfast
import spherical_functions as sf
import scri
import pytest

ABD = scri.AsymptoticBondiData
abd = scri.asymptotic_bondi_data


def construct_and_validate(modifier, validator, ell_max=8):
    time = np.linspace(-100, 100, num=2001)
    sigma, sigmadot, sigmaddot, psi2, psi1, psi0 = np.zeros((6, sf.LM_total_size(0, ell_max)), dtype=complex)
    modifier(sigma, sigmadot, sigmaddot, psi2, psi1, psi0)
    abd = ABD.from_initial_values(time, ell_max, sigma, sigmadot, sigmaddot, psi2, psi1, psi0)
    validator(abd)
    return True


def check_modes(modes, nonzero_ℓm):
    import numpy as np
    import spherical_functions as sf

    non_zero_indices = np.array([modes.index(ℓ, m) for ℓ, m in nonzero_ℓm], dtype=int)
    zero_indices = np.array(list(set(range(sf.LM_total_size(0, modes.ell_max))) - set(non_zero_indices)), dtype=int)
    assert not np.any(modes[..., zero_indices]), f"nonzero values among indices {zero_indices}"
    for non_zero_index in non_zero_indices:
        assert np.any(modes[..., non_zero_index]), f"no nonzero values at index {non_zero_index}"


def test0():
    """Set only terms that are forbidden by spin weights; ensure all zeros"""

    def modifier(sigma, sigmadot, sigmaddot, psi2, psi1, psi0):
        psi0[: sf.LM_index(1, 1, 0)] = 1.234
        psi1[0] = 0.123
        sigma[: sf.LM_index(1, 1, 0)] = 0.567
        sigmadot[: sf.LM_index(1, 1, 0)] = 0.678
        sigmaddot[: sf.LM_index(1, 1, 0)] = 0.789

    def validator(abd):
        check_modes(abd.psi0, [])
        check_modes(abd.psi1, [])
        check_modes(abd.psi2, [])
        check_modes(abd.psi3, [])
        check_modes(abd.psi4, [])
        check_modes(abd.sigma, [])
        assert np.max(np.abs(abd.bondi_violation_norms)) == 0.0

    construct_and_validate(modifier, validator)


def test1():
    """Add ℓ=0 term to ψ₂ initial value

    Ensures that first terms of ψ̇₁u = ðψ₂ + 2σψ₃ and ψ̇₀ = ðψ₁ + 3σψ₂ don't get excited

    """

    def modifier(sigma, sigmadot, sigmaddot, psi2, psi1, psi0):
        # Nonsensical values that should have no effect
        psi0[: sf.LM_index(1, 1, 0)] = 1.234
        psi1[0] = 0.123
        sigma[: sf.LM_index(1, 1, 0)] = 0.567
        sigmadot[: sf.LM_index(1, 1, 0)] = 0.678
        sigmaddot[: sf.LM_index(1, 1, 0)] = 0.789
        # Actual values that should carry through
        psi2[sf.LM_index(0, 0, 0)] = 0.234

    def validator(abd):
        assert np.all(abd.psi2[..., 0] == 0.234)
        check_modes(abd.psi0, [])
        check_modes(abd.psi1, [])
        check_modes(abd.psi2, [[0, 0]])
        check_modes(abd.psi3, [])
        check_modes(abd.psi4, [])
        check_modes(abd.sigma, [])
        assert np.max(np.abs(abd.bondi_violation_norms)) == 0.0

    construct_and_validate(modifier, validator, ell_max=3)


def test2():
    """Add ℓ=1 term to ψ₂ initial value

    Checks first term of ψ̇₁ = ðψ₂ + 2σψ₃

    """

    def modifier(sigma, sigmadot, sigmaddot, psi2, psi1, psi0):
        # Nonsensical values that should have no effect
        psi0[: sf.LM_index(1, 1, 0)] = 1.234
        psi1[0] = 0.123
        sigma[: sf.LM_index(1, 1, 0)] = 0.567
        sigmadot[: sf.LM_index(1, 1, 0)] = 0.678
        sigmaddot[: sf.LM_index(1, 1, 0)] = 0.789
        # Actual values that should carry through
        psi2[sf.LM_index(0, 0, 0)] = 0.234
        psi2[sf.LM_index(1, -1, 0)] = 0.345

    def validator(abd):
        check_modes(abd.psi0, [])
        check_modes(abd.psi1, [[1, -1], [1, 1]])
        check_modes(abd.psi2, [[0, 0], [1, -1], [1, 1]])
        check_modes(abd.psi3, [])
        check_modes(abd.psi4, [])
        check_modes(abd.sigma, [])
        assert np.max(np.abs(abd.bondi_violation_norms)) < 1e-13

    construct_and_validate(modifier, validator, ell_max=4)


def test3():
    """Add ℓ=2 term to ψ₂ initial value

    Checks first term of ψ̇₀ = ðψ₁ + 3σψ₂

    """

    def modifier(sigma, sigmadot, sigmaddot, psi2, psi1, psi0):
        # Nonsensical values that should have no effect
        psi0[: sf.LM_index(1, 1, 0)] = 1.234
        psi1[0] = 0.123
        sigma[: sf.LM_index(1, 1, 0)] = 0.567
        sigmadot[: sf.LM_index(1, 1, 0)] = 0.678
        sigmaddot[: sf.LM_index(1, 1, 0)] = 0.789
        # Actual values that should carry through
        psi2[sf.LM_index(0, 0, 0)] = 0.234
        psi2[sf.LM_index(1, -1, 0)] = 0.345
        psi2[sf.LM_index(2, -2, 0)] = 0.456

    def validator(abd):
        assert np.all(abd.psi2[..., 0] == 0.234)
        check_modes(abd.psi0, [[2, -2], [2, 2]])
        check_modes(abd.psi1, [[1, -1], [1, 1], [2, -2], [2, 2]])
        check_modes(abd.psi2, [[0, 0], [1, -1], [1, 1], [2, -2], [2, 2]])
        check_modes(abd.psi3, [])
        check_modes(abd.psi4, [])
        check_modes(abd.sigma, [])
        assert np.max(np.abs(abd.bondi_violation_norms)) < 4e-11

    construct_and_validate(modifier, validator, ell_max=4)


def test4():
    """Add nonzero constant term to σ

    Checks first term of ψ̇₁ = ðψ₂ + 2σψ₃ and second term of ψ̇₀ = ðψ₁ + 3σψ₂

    After satisfaction of the reality condition on the mass aspect, ψ₂ has nonzero modes in
    {(0, 0), (2, -2), (2, 2)}, so ψ₁ should have nonzero modes in {(2, -2), (2, 2)}.  Here,
    σ has only the nonzero mode (2, 2).  Thus the product σψ₂ should result in nonzero
    modes in {(2, -2), (2, 0), (2, 2), (3, 0), (4, 0), (4, 4)}.

    """

    def modifier(sigma, sigmadot, sigmaddot, psi2, psi1, psi0):
        # Nonsensical values that should have no effect
        psi0[: sf.LM_index(1, 1, 0)] = 1.234
        psi1[0] = 0.123
        sigma[: sf.LM_index(1, 1, 0)] = 0.567
        sigmadot[: sf.LM_index(1, 1, 0)] = 0.678
        sigmaddot[: sf.LM_index(1, 1, 0)] = 0.789
        # Actual values that should carry through
        psi2[0] = 0.234
        sigma[sf.LM_index(2, 2, 0)] = 0.5678

    def validator(abd):
        check_modes(abd.psi0, [[2, -2], [2, 0], [2, 2], [3, 0], [4, 0], [4, 4]])
        check_modes(abd.psi1, [[2, -2], [2, 2]])
        check_modes(abd.psi2, [[0, 0], [2, -2], [2, 2]])
        check_modes(abd.psi3, [])
        check_modes(abd.psi4, [])
        check_modes(abd.sigma, [[2, 2]])
        assert np.max(np.abs(abd.bondi_violation_norms)) <= 2e-10

    construct_and_validate(modifier, validator, ell_max=6)


def test5():
    """Add nonzero ℓ=2 term to σ̇

    Checks second term of ψ̇₁ = ðψ₂ + 2σψ₃

    """

    def modifier(sigma, sigmadot, sigmaddot, psi2, psi1, psi0):
        # Nonsensical values that should have no effect
        psi0[: sf.LM_index(1, 1, 0)] = 1.234
        psi1[0] = 0.123
        sigma[: sf.LM_index(1, 1, 0)] = 0.567
        sigmadot[: sf.LM_index(1, 1, 0)] = 0.678
        sigmaddot[: sf.LM_index(1, 1, 0)] = 0.789
        # Actual values that should carry through
        sigma[sf.LM_index(2, 2, 0)] = 0.5678
        sigmadot[sf.LM_index(2, 2, 0)] = 0.6789

    def validator(abd):
        check_modes(abd.psi0, [[2, -2], [2, 0], [2, 2], [3, 0], [4, 0], [4, 4]])
        check_modes(abd.psi1, [[1, 0], [2, -2], [2, 0], [2, 2], [3, 0], [4, 0]])
        check_modes(abd.psi2, [[2, -2], [2, 2]])
        check_modes(abd.psi3, [[2, -2]])
        check_modes(abd.psi4, [])
        check_modes(abd.sigma, [[2, 2]])
        assert np.max(np.abs(abd.bondi_violation_norms)) <= 7e-9

    construct_and_validate(modifier, validator, ell_max=6)


def test6():
    """Add nonzero ℓ=2 term to σ̈

    Checks second term of ψ̇₂ = ðψ₃ + σψ₄

    """

    def modifier(sigma, sigmadot, sigmaddot, psi2, psi1, psi0):
        # Nonsensical values that should have no effect
        psi0[: sf.LM_index(1, 1, 0)] = 1.234
        psi1[0] = 0.123
        sigma[: sf.LM_index(1, 1, 0)] = 0.567
        sigmadot[: sf.LM_index(1, 1, 0)] = 0.678
        sigmaddot[: sf.LM_index(1, 1, 0)] = 0.789
        # Actual values that should carry through
        sigmaddot[sf.LM_index(2, 2, 0)] = 0.1 / 10_000 ** 2  # 10_000 = max(time)**2

    def validator(abd):
        check_modes(abd.psi0, [[2, -2], [2, 0], [2, 2], [3, 0], [3, 2], [4, 0], [4, 2], [5, 2], [6, 2]])
        check_modes(abd.psi1, [[1, 0], [2, -2], [2, 0], [3, 0], [4, 0]])
        check_modes(abd.psi2, [[0, 0], [1, 0], [2, -2], [2, 0], [3, 0], [4, 0]])
        check_modes(abd.psi3, [[2, -2]])
        check_modes(abd.psi4, [[2, -2]])
        check_modes(abd.sigma, [[2, 2]])
        assert np.max(np.abs(abd.bondi_violation_norms)) <= 5e-8

    construct_and_validate(modifier, validator, ell_max=7)


def test7():
    """Test random values for all ℓ modes"""
    ell_max = 8
    np.random.seed(1234)

    def modifier(sigma, sigmadot, sigmaddot, psi2, psi1, psi0):
        sigma[:] = 0.01 * np.random.rand(*(sigma.shape[:-1] + (sigma.shape[-1] * 2,))).view(complex)
        sigmadot[:] = (0.01 / 100) * np.random.rand(*(sigma.shape[:-1] + (sigma.shape[-1] * 2,))).view(complex)
        sigmaddot[:] = (0.01 / 100 ** 2) * np.random.rand(*(sigma.shape[:-1] + (sigma.shape[-1] * 2,))).view(complex)

    def validator(abd):
        check_modes(abd.psi0, sf.LM_range(abs(abd.psi0.s), ell_max))
        check_modes(abd.psi1, sf.LM_range(abs(abd.psi1.s), ell_max))
        check_modes(abd.psi2, sf.LM_range(abs(abd.psi2.s), ell_max))
        check_modes(abd.psi3, sf.LM_range(2, ell_max))
        check_modes(abd.psi4, sf.LM_range(abs(abd.psi4.s), ell_max))
        check_modes(abd.sigma, sf.LM_range(abs(abd.sigma.s), ell_max))
        assert np.max(np.abs(abd.bondi_violation_norms)) <= 4.5e-6

    construct_and_validate(modifier, validator, ell_max=ell_max)
