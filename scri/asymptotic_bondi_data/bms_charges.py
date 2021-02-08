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

        M = -Re(\\psi_2 + \\sigma * \\dot{\\bar{\\sigma}})

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
        return -(self.psi2 + self.sigma.multiply(self.sigma.bar.dot, truncator=truncate_ell))
    elif truncate_ell:
        return -(
            self.psi2.truncate_ell(truncate_ell)
            + self.sigma.multiply(self.sigma.bar.dot, truncator=lambda tup: truncate_ell)
        ).real
    else:
        return -(self.psi2 + self.sigma * self.sigma.bar.dot).real


def charge_vector_from_aspect(charge):
    """Output the ell<=1 modes of a BMS charge aspect as the charge four-vector."""
    four_vector = np.empty(charge.shape, dtype=float)
    four_vector[..., 0] = charge[..., 0].real
    four_vector[..., 1] = (charge[..., 1] - charge[..., 3]).real / sqrt(6)
    four_vector[..., 2] = (charge[..., 1] + charge[..., 3]).imag / sqrt(6)
    four_vector[..., 3] = charge[..., 2].real / sqrt(3)
    return four_vector / np.sqrt(4 * np.pi)


def bondi_four_momentum(self):
    """Compute the Bondi four-momentum of the AsymptoticBondiData."""
    ell_max = 1  # Compute only the parts we need, ell<=1
    charge_aspect = self.mass_aspect(ell_max).view(np.ndarray)
    return charge_vector_from_aspect(charge_aspect)
