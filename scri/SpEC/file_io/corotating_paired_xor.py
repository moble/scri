# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

import warnings
import os
import numpy as np
import quaternion
from quaternion.numba_wrapper import jit, xrange
import spherical_functions as sf
from .. import (WaveformModes, FrameNames, DataType, DataNames, UnknownDataType, h, hdot, psi4, psi3, psi2, psi1, psi0)
from sxs.metadata import Metadata


def save(w, file_name=None, L2norm_fractional_tolerance=1e-10, log_frame=None, compress=True):
    import tempfile
    import contextlib
    import pathlib
    import json
    import numpy
    import scipy
    import h5py
    import sxs
    import scri
    from ...utilities import xor_timeseries

    compression_options = {
        'compression': 'gzip',
        'compression_opts': 9,
        'shuffle': True,
        'fletcher32': True,
    } if compress else {}

    if L2norm_fractional_tolerance == 0.0:
        log_frame = quaternion.as_float_array(np.log(w.frame))[:, 1:]
    else:
        # We need this storage anyway, so let's just make a copy and work in-place
        w = w.copy()
        if log_frame is not None:
            log_frame = log_frame.copy()

        # Ensure waveform is in corotating frame
        if w.frameType == scri.Inertial:
            try:
                initial_time = w.t[0]
                relaxation_time = w.metadata.relaxation_time
                max_norm_time = w.max_norm_time()
                z_alignment_region = ((relaxation_time - initial_time) / (max_norm_time - initial_time), 0.95)
            except:
                z_alignment_region = (0.1, 0.95)
            w, log_frame = w.to_corotating_frame(
                tolerance=1e-10, z_alignment_region=z_alignment_region, truncate_log_frame=True
            )
            log_frame = log_frame[:, 1:]
        if w.frameType != scri.Corotating:
            raise ValueError("Frame type of input waveform must be 'Corotating' or 'Inertial'; "
                             f"it is {w.frame_type_string}")

        # Convert mode structure to conjugate pairs
        w.convert_to_conjugate_pairs()

        # Set bits below the desired significance level to 0
        w.truncate(tol=L2norm_fractional_tolerance)

        # Compute log(frame)
        if log_frame is None:
            log_frame = quaternion.as_float_array(np.log(w.frame))[:, 1:]
            power_of_2 = 2 ** (-np.floor(np.log2(L2norm_fractional_tolerance/10))).astype('int')
            log_frame = np.round(log_frame * power_of_2) / power_of_2

        # Change -0.0 to 0.0 (~.5% compression for non-precessing systems)
        w.t += 0.0
        w.data += 0.0
        log_frame += 0.0

        # XOR successive instants in time
        xor_timeseries(w.t)
        xor_timeseries(w.data)
        xor_timeseries(log_frame)

    # Make sure we have a place to keep all this
    with contextlib.ExitStack() as context:
        if file_name is None:
            temp_dir = context.enter_context(tempfile.TemporaryDirectory())
            h5_file_name = Path(f'{temp_dir}') / 'test.h5'
        else:
            print(f'Saving H5 to "{h5_file_name}"')

        # Write the H5 file
        with h5py.File(h5_file_name, 'w') as f:
            f.attrs['sxs_format'] = 'corotating_paired_xor'
            warnings.warn('sxs_format is being set to "corotating_paired_xor"')
            f.create_dataset('time', data=w.t.view(np.uint64), **compression_options)
            f.create_dataset('modes', data=w.data.view(np.uint64), chunks=(w.n_times, 1), **compression_options)
            f['modes'].attrs['ell_min'] = w.ell_min
            f['modes'].attrs['ell_max'] = w.ell_max
            if log_frame.size > 1:
                f.create_dataset('log_frame', data=log_frame.view(np.uint64), chunks=(w.n_times, 1), **compression_options)

        size = os.stat(file_name).st_size
        print(f'Output H5 file size: {size:_} B')

        # Write the corresponding JSON file
        json_file_name = h5_file_name.with_suffix('.json')
        warnings.warn('Incomplete JSON data; work in progress')
        json_data = {
            'sxs_format': 'corotating_paired_xor',
            'data_info': {
                'data_type': w.data_type_string,
                'spin_weight': int(w.spin_weight),
                'ell_min': int(w.ell_min),
                'ell_max': int(w.ell_max),
            },
            'transformations': {
                # 'boost_velocity': [],
                # 'translation': [],
                'truncation': L2norm_fractional_tolerance,
            },
            'version_info': {
                # 'SpEC': [],
                'numpy': numpy.__version__,
                'scipy': scipy.__version__,
                'h5py': h5py.__version__,
                'sxs': sxs.__version__,
                'scri': scri.__version__,
            },
            'validation': {
                'h5_file_size': size,
                'n_times': w.n_times,
                # 'fletcher32': {'time': [], 'modes': [], 'log_frame': []}
            }
        }
        print(f'Saving JSON to "{json_file_name}"')
        with json_file_name.open('w') as f:
            json.dump(json_data, f, indent=2, separators=(',', ': '), ensure_ascii=True)

    return w


def load(file_name):
    import pathlib
    import json
    import h5py
    import scri
    from ...utilities import xor_timeseries_reverse

    h5_file_name = pathlib.Path(file_name).expanduser().resolve().with_suffix('.h5')
    json_file_name = h5_file_name.with_suffix('.json')

    with open(json_file_name, 'r') as f:
        json_data = json.load(f)
        dataType = scri.DataType[scri.DataNames.index(json_data['data_info']['data_type'])]

    with h5py.File(h5_file_name, 'r') as f:
        sxs_format = f.attrs['sxs_format']
        assert sxs_format in ['corotating_paired_xor',]
        time = f['time'][:].view(np.float64)
        modes = f['modes'][:].view(np.complex128)
        ell_min = f['modes'].attrs['ell_min']
        ell_max = f['modes'].attrs['ell_max']
        if 'log_frame' in f:
            log_frame = f['log_frame'][:].view(np.float64)
        else:
            log_frame = np.empty((0, 3), dtype=np.float64)

    xor_timeseries_reverse(time)
    xor_timeseries_reverse(modes)
    xor_timeseries_reverse(log_frame)
    frame = np.exp(quaternion.as_quat_array(np.insert(log_frame, 0, 0.0, axis=1)))

    w = WaveformModes(
        t=time, frame=frame, data=modes,
        frameType=scri.Corotating, dataType=dataType,
        m_is_scaled_out=True, r_is_scaled_out=True,
        ell_min=ell_min, ell_max=ell_max
    )
    w.convert_from_conjugate_pairs()

    return w
