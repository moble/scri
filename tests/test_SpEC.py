# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

from __future__ import print_function, division, absolute_import

import pytest
import os.path
import numpy as np
import h5py
import scri
import scri.SpEC


@pytest.mark.skipif(not os.path.exists('../SpEC/samples/rhOverM_Asymptotic_GeometricUnits.h5'),
                    reason="requires SpEC/samples/*.h5")
def test_file_io():
    """Test file I/O with a round-trip and compare H5 files"""
    w = scri.SpEC.read_from_h5('../SpEC/samples/rhOverM_Asymptotic_GeometricUnits.h5/Extrapolated_N2.dir')
    scri.SpEC.write_to_h5(w, '../SpEC/samples/Asymptotic_GeometricUnits_test.h5/Extrapolated_N2.dir')

    f1 = h5py.File('../SpEC/samples/rhOverM_Asymptotic_GeometricUnits.h5')
    f2 = h5py.File('../SpEC/samples/rhOverM_Asymptotic_GeometricUnits_test.h5')
    w1 = f1['Extrapolated_N2.dir']
    w2 = f2['Extrapolated_N2.dir']

    # Check top-level attributes
    assert list(w1.attrs.items())[1:] == list(w2.attrs.items())[1:]  # 0 attribute is waveform output version

    # Check that original history is contained (with extra comment characters) in second history
    assert ('# ' + '\n# '.join(w1['History.txt'][()].decode().split('\n'))) in w2['History.txt'][()]

    # Now check each mode from the original
    for mode in list(w1):
        if mode.startswith('Y'):
            assert list(w1[mode].attrs.items()) == list(w2[mode].attrs.items())
            assert np.array_equal(w1[mode][:], w2[mode][:])