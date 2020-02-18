import math
import numpy as np
import quaternion
import spinsfast
import spherical_functions as sf
import scri
import pytest

ABD = scri.data.AsymptoticBondiData


def kerr_schild(mass, spin, ell_max=8):
    psi2 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)
    psi1 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)
    psi0 = np.zeros(sf.LM_total_size(0, ell_max), dtype=complex)
    
    psi2[0] = (mass) * (-math.sqrt(8))
    psi1[2] = (3j * spin / 2) * (np.sqrt((8/3) * np.pi))
    psi0[6] = (3 * spin**2 / mass / 2) * (np.sqrt((32/15) * np.pi))

    return psi2, psi1, psi0


def test_abd_schwarzschild():
    ell_max = 8
    u = np.linspace(0, 100, num=5000)
    psi2, psi1, psi0 = kerr_schild(1.0, 0.0, ell_max)
    abd = ABD.from_initial_values(u, ell_max=ell_max, psi2=psi2)
    expected_four_momentum = np.zeros((u.size, 4), dtype=float)
    expected_four_momentum[..., 0] = 1.0
    computed_four_momentum = abd.bondi_four_momentum
    assert np.array_equal(computed_four_momentum, expected_four_momentum)
    #print('Bondi-violation norms', abd.bondi_violation_norms)
