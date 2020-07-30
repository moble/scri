# Copyright (c) 2018, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import numpy as np
import quaternion
from numpy import *
import pytest
import scri

from conftest import random_waveform


def test_parity_projections():
    for dataType in [scri.psi0, scri.psi1, scri.psi2, scri.psi3, scri.psi4, scri.h]:
        w = random_waveform(dataType=dataType)
        ## x
        x = w.x_parity_symmetric_part
        assert np.array_equal(x.data, x.x_parity_conjugate.data)
        assert np.array_equal(x.data, x.x_parity_symmetric_part.data)
        assert np.array_equal(np.zeros_like(x.data), x.x_parity_antisymmetric_part.data)
        assert np.array_equal(np.zeros_like(x.t), x.x_parity_violation_squared)
        x = w.x_parity_antisymmetric_part
        assert np.array_equal(x.data, -x.x_parity_conjugate.data)
        assert np.array_equal(x.data, x.x_parity_antisymmetric_part.data)
        assert np.array_equal(np.zeros_like(x.data), x.x_parity_symmetric_part.data)
        assert np.array_equal(x.norm(), x.x_parity_violation_squared)
        ## y
        y = w.y_parity_symmetric_part
        assert np.array_equal(y.data, y.y_parity_conjugate.data)
        assert np.array_equal(y.data, y.y_parity_symmetric_part.data)
        assert np.array_equal(np.zeros_like(y.data), y.y_parity_antisymmetric_part.data)
        assert np.array_equal(np.zeros_like(y.t), y.y_parity_violation_squared)
        y = w.y_parity_antisymmetric_part
        assert np.array_equal(y.data, -y.y_parity_conjugate.data)
        assert np.array_equal(y.data, y.y_parity_antisymmetric_part.data)
        assert np.array_equal(np.zeros_like(y.data), y.y_parity_symmetric_part.data)
        assert np.array_equal(y.norm(), y.y_parity_violation_squared)
        ## z
        z = w.z_parity_symmetric_part
        assert np.array_equal(z.data, z.z_parity_conjugate.data)
        assert np.array_equal(z.data, z.z_parity_symmetric_part.data)
        assert np.array_equal(np.zeros_like(z.data), z.z_parity_antisymmetric_part.data)
        assert np.array_equal(np.zeros_like(z.t), z.z_parity_violation_squared)
        z = w.z_parity_antisymmetric_part
        assert np.array_equal(z.data, -z.z_parity_conjugate.data)
        assert np.array_equal(z.data, z.z_parity_antisymmetric_part.data)
        assert np.array_equal(np.zeros_like(z.data), z.z_parity_symmetric_part.data)
        assert np.array_equal(z.norm(), z.z_parity_violation_squared)
        ## xyz
        xyz = w.parity_symmetric_part
        assert np.array_equal(xyz.data, xyz.parity_conjugate.data)
        assert np.array_equal(xyz.data, xyz.parity_symmetric_part.data)
        assert np.array_equal(np.zeros_like(xyz.data), xyz.parity_antisymmetric_part.data)
        assert np.array_equal(np.zeros_like(xyz.t), xyz.parity_violation_squared)
        xyz = w.parity_antisymmetric_part
        assert np.array_equal(xyz.data, -xyz.parity_conjugate.data)
        assert np.array_equal(xyz.data, xyz.parity_antisymmetric_part.data)
        assert np.array_equal(np.zeros_like(xyz.data), xyz.parity_symmetric_part.data)
        assert np.array_equal(xyz.norm(), xyz.parity_violation_squared)
