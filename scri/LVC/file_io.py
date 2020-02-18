# Copyright (c) 2018, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

import warnings
import numpy as np
import spherical_functions as sf
from .. import (WaveformModes, Inertial, h)

def read_from_h5(file_name, **kwargs):
    """Read data from an H5 file in LVC format"""
    import re
    import h5py
    from scipy.interpolate import InterpolatedUnivariateSpline as Spline

    phase_re = re.compile('phase_l(?P<ell>.*)_m(?P<m>.*)')
    amp_re = re.compile('amp_l(?P<ell>.*)_m(?P<m>.*)')
    
    with h5py.File(file_name) as f:
        t = f['NRtimes'][:]
        ell_m = np.array([[int(match['ell']), int(match['m'])] for key in f for match in [phase_re.match(key)] if match])
        ell_min = np.min(ell_m[:, 0])
        ell_max = np.max(ell_m[:, 0])
        data = np.empty((t.size, sf.LM_total_size(ell_min, ell_max)), dtype=complex)
        for ell in range(ell_min, ell_max+1):
            for m in range(-ell, ell+1):
                amp = Spline(f['amp_l{0}_m{1}/X'.format(ell, m)][:],
                             f['amp_l{0}_m{1}/Y'.format(ell, m)][:],
                             k=int(f['amp_l{0}_m{1}/deg'.format(ell, m)][()]))(t)
                phase = Spline(f['phase_l{0}_m{1}/X'.format(ell, m)][:],
                               f['phase_l{0}_m{1}/Y'.format(ell, m)][:],
                               k=int(f['phase_l{0}_m{1}/deg'.format(ell, m)][()]))(t)
                data[:, sf.LM_index(ell, m, ell_min)] = amp * np.exp(1j * phase)
        if 'auxiliary-info' in f and 'history.txt' in f['auxiliary-info']:
            history = ("### " + f['auxiliary-info/history.txt'][()].decode().replace('\n', '\n### ')).split('\n')
        else:
            history = [""]
        constructor_statement = "scri.LVC.read_from_h5('{0}')".format(file_name)
        w = WaveformModes(t=t, data=data, ell_min=ell_min, ell_max=ell_max,
                          frameType=Inertial, dataType=h,
                          history=history, constructor_statement=constructor_statement,
                          r_is_scaled_out=True, m_is_scaled_out=True)

    return w
