# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

import warnings
import os
import numpy as np
import quaternion
from quaternion.numba_wrapper import jit, xrange
import spherical_functions as sf
from ... import WaveformModes, DataType, DataNames


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

    # Make sure that we can understand the file_name and create the directory
    if file_name is None:
        # We'll just be creating a temp directory below, to check
        warnings.warn(
            'Input `file_name` is None.  Running in temporary directory.\n'
            'Note that this option is mostly for debugging purposes.'
        )
    else:
        h5_path = pathlib.Path(file_name).expanduser().resolve().with_suffix('.h5')
        if not h5_path.parent.exists():
            h5_path.parent.mkdir(parents=True)

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
            h5_path = pathlib.Path(f'{temp_dir}') / 'test.h5'
        else:
            print(f'Saving H5 to "{h5_path}"')

        # Write the H5 file
        with h5py.File(h5_path, 'w') as f:
            f.attrs['sxs_format'] = 'corotating_paired_xor'
            warnings.warn('sxs_format is being set to "corotating_paired_xor"')
            f.create_dataset('time', data=w.t.view(np.uint64), chunks=(w.n_times,), **compression_options)
            f.create_dataset('modes', data=w.data.view(np.uint64), chunks=(w.n_times, 1), **compression_options)
            f['modes'].attrs['ell_min'] = w.ell_min
            f['modes'].attrs['ell_max'] = w.ell_max
            if log_frame.size > 1:
                f.create_dataset('log_frame', data=log_frame.view(np.uint64), chunks=(w.n_times, 1), **compression_options)

        h5_size = os.stat(h5_path).st_size
        if file_name is None:
            print(f'Output H5 file size: {h5_size:_} B')

        # Write the corresponding JSON file
        json_path = h5_path.with_suffix('.json')
        json_data = {
            'sxs_format': 'corotating_paired_xor',
            'data_info': {
                'data_type': w.data_type_string,
                'spin_weight': int(w.spin_weight),
                'ell_min': int(w.ell_min),
                'ell_max': int(w.ell_max),
            },
            'transformations': {
                'truncation': L2norm_fractional_tolerance,
                # see below for 'boost_velocity'
                # see below for 'space_translation'
            },
            'version_info': {
                'numpy': numpy.__version__,
                'scipy': scipy.__version__,
                'h5py': h5py.__version__,
                'quaternion': quaternion.__version__,
                'spherical_functions': sf.__version__,
                'scri': scri.__version__,
                'sxs': sxs.__version__,
                # see below 'spec_version_hist'
            },
            'validation': {
                'h5_file_size': h5_size,
                'n_times': w.n_times,
                # 'fletcher32': {'time': [], 'modes': [], 'log_frame': []}
            }
        }
        if hasattr(w, 'boost_velocity'):
            json_data['transformations']['boost_velocity'] = w.boost_velocity.tolist()
        if hasattr(w, 'space_translation'):
            json_data['transformations']['space_translation'] = w.space_translation.tolist()
        if hasattr(w, 'version_hist'):
            json_data['version_info']['spec_version_history'] = w.version_hist
        if file_name is not None:
            print(f'Saving JSON to "{json_path}"')
        with json_path.open('w') as f:
            json.dump(json_data, f, indent=2, separators=(',', ': '), ensure_ascii=True)

    return w


def load(file_name):
    import pathlib
    import json
    import h5py
    import scri
    from ...utilities import xor_timeseries_reverse

    h5_path = pathlib.Path(file_name).expanduser().resolve().with_suffix('.h5')
    json_path = h5_path.with_suffix('.json')

    with open(json_path, 'r') as f:
        json_data = json.load(f)
        dataType = scri.DataType[scri.DataNames.index(json_data['data_info']['data_type'])]

    with h5py.File(h5_path, 'r') as f:
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
    w.json_data = json_data

    return w
