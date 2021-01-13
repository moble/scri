# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

### NOTE: The functions in this file are intended purely for inclusion in the AsymptoticBondData
### class.  In particular, they assume that the first argument, `self` is an instance of
### AsymptoticBondData.  They should probably not be used outside of that class.

import numpy as np
from math import sqrt, pi


def mass_aspect(self, truncate_ell=max):
    """Compute the Bondi mass aspect of the AsymptoticBondiData

    The Bondi mass aspect is given by

        \\Psi = \\psi_2 + \\eth \\eth \\bar{\\sigma} + \\sigma * \\dot{\\bar{\\sigma}}

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
        any of the individual Modes objects in the equation for \\Psi above -- but not the
        product.

    """
    if callable(truncate_ell):
        return self.psi2 + self.sigma.bar.eth_GHP.eth_GHP + self.sigma.multiply(self.sigma.bar.dot, truncator=truncate_ell)
    elif truncate_ell:
        return (
            self.psi2.truncate_ell(truncate_ell)
            + self.sigma.bar.eth_GHP.eth_GHP.truncate_ell(truncate_ell)
            + self.sigma.multiply(self.sigma.bar.dot, truncator=lambda tup: truncate_ell)
        )
    else:
        return self.psi2 + self.sigma.bar.eth_GHP.eth_GHP + self.sigma * self.sigma.bar.dot


def bondi_four_momentum(self):
    """Compute the Bondi four-momentum of the AsymptoticBondiData"""
    import spherical_functions as sf

    P_restricted = -self.mass_aspect(1).view(np.ndarray) / sqrt(4 * pi)  # Compute only the parts we need, ell<=1
    four_momentum = np.empty(P_restricted.shape, dtype=float)
    four_momentum[..., 0] = P_restricted[..., 0].real
    four_momentum[..., 1] = (P_restricted[..., 3] - P_restricted[..., 1]).real / sqrt(6)
    four_momentum[..., 2] = (1j * (P_restricted[..., 3] + P_restricted[..., 1])).real / sqrt(6)
    four_momentum[..., 3] = -P_restricted[..., 2].real / sqrt(3)
    return four_momentum
