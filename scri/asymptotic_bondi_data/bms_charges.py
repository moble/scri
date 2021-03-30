# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

### NOTE: The functions in this file are intended purely for inclusion in the AsymptoticBondData
### class.  In particular, they assume that the first argument, `self` is an instance of
### AsymptoticBondData.  They should probably not be used outside of that class.

import numpy as np
from math import sqrt, pi


def mass_aspect(self, truncate_ell=max):
    """Compute the Bondi mass aspect of the AsymptoticBondiData.

    The Bondi mass aspect is given by

        M = -ℜ{ψ₂ + σ ∂ₜσ̄}

    Note that the last term is a product between two fields.  If, for example, these both have
    ell_max=8, then their full product would have ell_max=16, meaning that we would go from
    tracking 81 modes to 289.  This shows that deciding how to truncate the output ell is
    important, which is why this function has the extra argument that it does.

    Parameters
    ==========
    truncate_ell: int, or callable [defaults to `max`]
        Determines how the ell_max value of the output is determined.  If an integer is passed,
        each term in the output is truncated to have at most that ell_max.  (In particular,
        terms that will not be used in the output are simply not computed, without incurring any
        errors due to aliasing.)  If a callable is passed, it is passed on to the
        spherical_functions.Modes.multiply method.  See that function's docstring for details.
        The default behavior will result in the output having ell_max equal to the largest of
        any of the individual Modes objects in the equation for M above -- but not the
        product.

    """
    if callable(truncate_ell):
        return -(self.psi2 + self.sigma.multiply(self.sigma.bar.dot, truncator=truncate_ell)).real
    elif truncate_ell:
        return -(
            self.psi2.truncate_ell(truncate_ell)
            + self.sigma.multiply(self.sigma.bar.dot, truncator=lambda tup: truncate_ell)
        ).real
    else:
        return -(self.psi2 + self.sigma * self.sigma.bar.dot).real


def charge_vector_from_aspect(charge):
    """Output the ell<=1 modes of a BMS charge aspect as the charge four-vector.

    Considering the aspect as a function a(θ, ϕ), we express the corresponding
    four-vector in terms of Cartesian components as

      vᵝ = (1/4π) ∫ ℜ{a} (tᵝ + rᵝ) dΩ

    where the integral is taken over the 2-sphere, and tᵝ + rᵝ has components
    (1, sinθ cosϕ, sinθ sinϕ, cosθ).

    """
    four_vector = np.empty(charge.shape, dtype=float)
    four_vector[..., 0] = charge[..., 0].real
    four_vector[..., 1] = (charge[..., 1] - charge[..., 3]).real / sqrt(6)
    four_vector[..., 2] = (charge[..., 1] + charge[..., 3]).imag / sqrt(6)
    four_vector[..., 3] = charge[..., 2].real / sqrt(3)
    return four_vector / np.sqrt(4 * np.pi)


def bondi_rest_mass(self):
    """Compute the rest mass from the Bondi four-momentum"""
    four_momentum = self.bondi_four_momentum()
    rest_mass = np.sqrt(four_momentum[:, 0] ** 2 - np.sum(four_momentum[:, 1:] ** 2, axis=1))
    return rest_mass


def bondi_four_momentum(self):
    """Compute the Bondi four-momentum

    This is just the ell<2 component of the mass aspect, expressed as a four-vector.

    """
    ell_max = 1  # Compute only the parts we need, ell<=1
    charge_aspect = self.mass_aspect(ell_max).view(np.ndarray)
    return charge_vector_from_aspect(charge_aspect)


def bondi_angular_momentum(self, output_dimensionless=False):
    """Compute the total Bondi angular momentum vector

        i (ψ₁ + σ ðσ̄)

    See Eq. (8) of Dray (1985) iopscience.iop.org/article/10.1088/0264-9381/2/1/002

    """
    ell_max = 1  # Compute only the parts we need, ell<=1
    charge_aspect = (
        1j
        * (self.psi1.truncate_ell(ell_max) + self.sigma.multiply(self.sigma.bar.eth_GHP, truncator=lambda tup: ell_max))
    ).view(np.ndarray)
    return charge_vector_from_aspect(charge_aspect)[:, 1:]


def bondi_dimensionless_spin(self):
    """Compute the dimensionless Bondi spin vector"""
    N = self.bondi_boost_charge()
    J = self.bondi_angular_momentum()
    P = self.bondi_four_momentum()
    M_sqr = (P[:, 0] ** 2 - np.sum(P[:, 1:] ** 2, axis=1))[:, np.newaxis]
    v = P[:, 1:] / (P[:, 0])[:, np.newaxis]
    v_norm = np.linalg.norm(v, axis=1)
    # To prevent dividing by zero, we compute the normalized velocity vhat ONLY at
    # timesteps with a non-zero velocity.
    vhat = v.copy()
    t_idx = v_norm != 0  # Get the indices for timesteps with non-zero velocity
    vhat[t_idx] = v[t_idx] / v_norm[t_idx, np.newaxis]
    gamma = (1 / np.sqrt(1 - v_norm ** 2))[:, np.newaxis]
    J_dot_vhat = np.einsum("ij,ij->i", J, vhat)[:, np.newaxis]
    spin_charge = (gamma * (J + np.cross(v, N)) - (gamma - 1) * J_dot_vhat * vhat) / M_sqr
    return spin_charge


def bondi_boost_charge(self):
    """Compute the Bondi boost charge vector

        - [ψ₁ + σ ðσ̄ + ½ð(σ σ̄) - t ð ℜ{ψ₂ + σ ∂ₜσ̄}]

    See Eq. (8) of Dray (1985) iopscience.iop.org/article/10.1088/0264-9381/2/1/002

    """
    ell_max = 1  # Compute only the parts we need, ell<=1
    charge_aspect = -(
        self.psi1.truncate_ell(ell_max)
        + self.sigma.multiply(self.sigma.bar.eth_GHP, truncator=lambda tup: ell_max)
        + 0.5 * (self.sigma.multiply(self.sigma.bar, truncator=lambda tup: ell_max)).eth_GHP
        - self.t
        * (
            self.psi2.truncate_ell(ell_max) + self.sigma.multiply(self.sigma.bar.dot, truncator=lambda tup: ell_max)
        ).real.eth_GHP
    ).view(np.ndarray)
    return charge_vector_from_aspect(charge_aspect)[:, 1:]


def bondi_CoM_charge(self):
    """Compute the center-of-mass charge vector

        Gⁱ = Nⁱ + t Pⁱ = - [ψ₁ + σ ðσ̄ + ½ð(σ σ̄)]

    where Nⁱ is the boost charge and Pⁱ is the momentum.  See Eq. (3.4) of arXiv:1912.03164.

    """
    ell_max = 1  # Compute only the parts we need, ell<=1
    charge_aspect = -(
        self.psi1.truncate_ell(ell_max)
        + self.sigma.multiply(self.sigma.bar.eth_GHP, truncator=lambda tup: ell_max)
        + 0.5 * (self.sigma.multiply(self.sigma.bar, truncator=lambda tup: ell_max)).eth_GHP
    ).view(np.ndarray)
    return charge_vector_from_aspect(charge_aspect)[:, 1:]


def supermomentum(self, supermomentum_def, **kwargs):
    """Compute the supermomentum

    This function allows for several different definitions of the
    supermomentum.  These differences only apply to ell > 1 modes,
    so they do not affect the Bondi four-momentum.  See
    Eqs. (7-9) in arXiv:1404.2475 for the different supermomentum
    definitions and links to further references.

    In the literature, there is an ambiguity of vocabulary.  When
    it comes to other BMS charges, we clearly distinuish between
    the "charge" and the "aspect".  However, the
    term "supermomentum" is used for both.  Accordingly, this
    function provides two ways to compute the supermomentum.

    1) By default, the supermomentum will be computed as

        Ψ = ψ₂ + σ ∂ₜσ̄ + f(θ, ϕ)

       (See below for the definitions of `f`.)

    2) By passing the option `integrated=True`, the supermomentum
       will instead be computed as

        Pₗₘ = - (1/4π) ∫ Ψ(θ, ϕ) Yₗₘ(θ, ϕ) dΩ

    Parameters
    ----------
    supermomentum_def : str
        The definition of the supermomentum to be computed.  One of the
        following (case-insensitive) options can be specified:
          * 'Bondi-Sachs' or 'BS' for f = 0
          * 'Moreschi' or 'M' for f = ð²σ̄
          * 'Geroch' or 'G' for f = ½ (ð²σ̄ - ð̄²σ)
          * 'Geroch-Winicour' or 'GW' for f = - ð̄²σ
    integrated : bool, optional
        If True, then return the integrated form of the supermomentum — see
        Eq. (6) in arXiv:1404.2475.  Default is False
    working_ell_max: int, optional
        The value of ell_max to be used to define the computation grid. The
        number of theta points and the number of phi points are set to
        2*working_ell_max+1. Defaults to 2*self.ell_max.

    Returns
    -------
    ModesTimeSeries

    """
    return_integrated = kwargs.pop("integrated", False)

    if supermomentum_def.lower() in ["bondi-sachs", "bs"]:
        supermomentum = self.psi2 + self.sigma.grid_multiply(self.sigma.bar.dot, **kwargs)
    elif supermomentum_def.lower() in ["moreschi", "m"]:
        supermomentum = self.psi2 + self.sigma.grid_multiply(self.sigma.bar.dot, **kwargs) + self.sigma.bar.eth_GHP.eth_GHP
    elif supermomentum_def.lower() in ["geroch", "g"]:
        supermomentum = (
            self.psi2
            + self.sigma.grid_multiply(self.sigma.bar.dot, **kwargs)
            + 0.5 * (self.sigma.bar.eth_GHP.eth_GHP - self.sigma.ethbar_GHP.ethbar_GHP)
        )
    elif supermomentum_def.lower() in ["geroch-winicour", "gw"]:
        supermomentum = self.psi2 + self.sigma.grid_multiply(self.sigma.bar.dot, **kwargs) - self.sigma.ethbar_GHP.ethbar_GHP
    else:
        raise ValueError(
            f"Supermomentum defintion '{supermomentum_def}' not recognized. Please choose one of "
            "the following options:\n"
            "  * 'Bondi-Sachs' or 'BS'\n"
            "  * 'Moreschi' or 'M'\n"
            "  * 'Geroch' or 'G'\n"
            "  * 'Geroch-Winicour' or 'GW'"
        )
    if return_integrated:
        return -0.5 * supermomentum.bar / np.sqrt(np.pi)
    else:
        return supermomentum
