# Copyright (c) 2019, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

import numpy as np
import quaternion
import spinsfast
from numpy import *
import pytest

from conftest import linear_waveform, constant_waveform, random_waveform


def silly_momentum_flux(h):
    """Compute momentum flux from waveform with a silly but simple method

    This function evaluates the momentum-flux formula quite literally.  The formula is

         dp       R**2  | dh |**2 ^
    ---------- = ------ | -- |    n
    dOmega  dt   16  pi | dt |

    Here, p and nhat are vectors and R is the distance to the source.  The input h is differentiated
    numerically to find modes of dh/dt, which are then used to construct the values of dh/dt on a
    grid.  At each point of that grid, the value of |dh/dt|**2 nhat is computed, which is then
    integrated over the sphere to arrive at dp/dt.  Note that this integration is accomplished by
    way of a spherical-harmonic decomposition; the ell=m=0 mode is multiplied by 2*sqrt(pi) to
    arrive at the result that would be achieved by integrating over the sphere.

    """
    import spinsfast
    hdot = h.data_dot
    zeros = np.zeros((hdot.shape[0], 4), dtype=hdot.dtype)
    data = np.concatenate((zeros, hdot), axis=1)  # Pad with zeros for spinsfast
    ell_min = 0
    ell_max = 2*h.ell_max + 1  # Maximum ell value required for nhat*|hdot|^2
    n_theta = 2*ell_max + 1
    n_phi = n_theta
    hdot_map = spinsfast.salm2map(data, h.spin_weight, h.ell_max, n_theta, n_phi)
    hdot_mag_squared_map = hdot_map * hdot_map.conjugate()

    theta = np.linspace(0.0, np.pi, num=n_theta, endpoint=True)
    phi = np.linspace(0.0, 2*np.pi, num=n_phi, endpoint=False)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    pdot = np.array([spinsfast.map2salm(hdot_mag_squared_map*n/(16*np.pi), 0, 0)[..., 0].real * (2*np.sqrt(np.pi))
                     for n in [x, y, z]]).T

    return pdot


def test_momentum_flux():
    import numpy as np
    import scri
    h = scri.sample_waveforms.fake_precessing_waveform(t_1=15_000.0)
    pdot1 = silly_momentum_flux(h)
    pdot2 = scri.momentum_flux(h)

    # diff = np.linalg.norm(pdot1-pdot2, axis=1)
    # ind = np.argmax(diff, axis=None)  # np.unravel_index(np.argmax(diff, axis=None), diff.shape)
    # print('Max diff norm:', diff[ind], pdot1[ind], pdot2[ind], ind)

    assert np.allclose(pdot1, pdot2, rtol=1e-13, atol=1e-13)
