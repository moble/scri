# Copyright (c) 2020, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

sxs_formats = [
    "rotating_paired_xor_multishuffle_bzip2",
]
default_shuffle_widths = (8, 8, 4, 4, 4, 2,) + (1,) * 34


def save(w, file_name=None, L2norm_fractional_tolerance=1e-10, log_frame=None, shuffle_widths=default_shuffle_widths):
    import sys
    import os
    import warnings
    import tempfile
    import contextlib
    import pathlib
    import bz2
    import json
    import numpy as np
    import scipy
    import h5py
    import quaternion
    import sxs
    import scri
    import spherical_functions as sf
    from scri.utilities import xor_timeseries as xor
    from sxs.utilities import md5checksum

    # Make sure that we can understand the file_name and create the directory
    if file_name is None:
        # We'll just be creating a temp directory below, to check
        warnings.warn(
            "\nInput `file_name` is None.  Running in temporary directory.\n"
            "Note that this option is mostly for debugging purposes."
        )
    else:
        h5_path = pathlib.Path(file_name).expanduser().resolve().with_suffix(".h5")
        if not h5_path.parent.exists():
            h5_path.parent.mkdir(parents=True)

    shuffle = scri.utilities.multishuffle(tuple(shuffle_widths))

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
            raise ValueError(
                "Frame type of input waveform must be 'Corotating' or 'Inertial'; " f"it is {w.frame_type_string}"
            )

        # Convert mode structure to conjugate pairs
        w.convert_to_conjugate_pairs()

        # Set bits below the desired significance level to 0
        w.truncate(tol=L2norm_fractional_tolerance)

        # Compute log(frame)
        if log_frame is None:
            log_frame = quaternion.as_float_array(np.log(w.frame))[:, 1:]
            power_of_2 = 2 ** (-np.floor(np.log2(L2norm_fractional_tolerance / 10))).astype("int")
            log_frame = np.round(log_frame * power_of_2) / power_of_2

        # Change -0.0 to 0.0 (~.5% compression for non-precessing systems)
        w.t += 0.0
        w.data += 0.0
        log_frame += 0.0

        # XOR successive instants in time
        xor(w.t)
        xor(w.data)
        xor(log_frame)

    # Make sure we have a place to keep all this
    with contextlib.ExitStack() as context:
        if file_name is None:
            temp_dir = context.enter_context(tempfile.TemporaryDirectory())
            h5_path = pathlib.Path(f"{temp_dir}") / "test.h5"
        else:
            print(f'Saving H5 to "{h5_path}"')

        # Write the H5 file
        with h5py.File(h5_path, "w") as f:
            if L2norm_fractional_tolerance != 0.0:
                f.attrs["sxs_format"] = f"{sxs_formats[0]}"
                f.attrs["n_times"] = w.n_times
                f.attrs["ell_min"] = w.ell_min
                f.attrs["ell_max"] = w.ell_max
                f.attrs["shuffle_widths"] = np.array(shuffle_widths, dtype=np.uint8)
                # warnings.warn(f'sxs_format is being set to "{sxs_formats[0]}"')
                data = np.void(
                    bz2.compress(
                        shuffle(w.t.view(np.uint64)).tobytes()
                        + shuffle(w.data.view(np.uint64).flatten("F")).tobytes()
                        + shuffle(log_frame.view(np.uint64).flatten("F")).tobytes()
                    )
                )
                f.create_dataset("data", data=data)
            else:
                compression_options = {
                    "compression": "gzip",
                    "compression_opts": 9,
                    "shuffle": True,
                }
                f.attrs["sxs_format"] = f"{sxs_formats[0]}"
                f.create_dataset("time", data=w.t.view(np.uint64), chunks=(w.n_times,), **compression_options)
                f.create_dataset("modes", data=w.data.view(np.uint64), chunks=(w.n_times, 1), **compression_options)
                f["modes"].attrs["ell_min"] = w.ell_min
                f["modes"].attrs["ell_max"] = w.ell_max
                if log_frame.size > 1:
                    f.create_dataset(
                        "log_frame", data=log_frame.view(np.uint64), chunks=(w.n_times, 1), **compression_options
                    )

        # Get some numbers for the JSON file
        h5_size = os.stat(h5_path).st_size
        if file_name is None:
            print(f"Output H5 file size: {h5_size:_} B")
        md5sum = md5checksum(h5_path)

        # Write the corresponding JSON file
        json_path = h5_path.with_suffix(".json")
        json_data = {
            "sxs_format": sxs_formats[0],
            "data_info": {
                "data_type": w.data_type_string,
                "spin_weight": int(w.spin_weight),
                "ell_min": int(w.ell_min),
                "ell_max": int(w.ell_max),
            },
            "transformations": {
                "truncation": L2norm_fractional_tolerance,
                # see below for 'boost_velocity'
                # see below for 'space_translation'
            },
            "version_info": {
                "python": sys.version,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "h5py": h5py.__version__,
                "quaternion": quaternion.__version__,
                "spherical_functions": sf.__version__,
                "scri": scri.__version__,
                "sxs": sxs.__version__,
                # see below 'spec_version_hist'
            },
            "validation": {"h5_file_size": h5_size, "n_times": w.n_times, "md5sum": md5sum},
        }
        if hasattr(w, "boost_velocity"):
            json_data["transformations"]["boost_velocity"] = w.boost_velocity.tolist()
        if hasattr(w, "space_translation"):
            json_data["transformations"]["space_translation"] = w.space_translation.tolist()
        if hasattr(w, "version_hist"):
            json_data["version_info"]["spec_version_history"] = w.version_hist
        if file_name is not None:
            print(f'Saving JSON to "{json_path}"')
        with json_path.open("w") as f:
            json.dump(json_data, f, indent=2, separators=(",", ": "), ensure_ascii=True)

    return w, log_frame


def load(file_name, ignore_validation=False, check_md5=True):
    import os
    import warnings
    import pathlib
    import bz2
    import json
    import numpy as np
    import h5py
    import quaternion
    import scri
    from scri.utilities import xor_timeseries_reverse as unxor
    from sxs.utilities import md5checksum

    def invalid(message):
        if ignore_validation:
            warnings.warn(message)
        else:
            raise ValueError(message)

    h5_path = pathlib.Path(file_name).expanduser().resolve().with_suffix(".h5")
    json_path = h5_path.with_suffix(".json")

    # This will be used for validation
    h5_size = os.stat(h5_path).st_size

    if not json_path.exists():
        invalid(f'\nJSON file "{json_path}" cannot be found, but is expected for this data format.')
        json_data = {}
    else:
        with open(json_path) as f:
            json_data = json.load(f)

        dataType = json_data.get("data_info", {}).get("data_type", "UnknownDataType")
        dataType = scri.DataType[scri.DataNames.index(dataType)]

        # Make sure this is our format
        sxs_format = json_data.get("sxs_format", "")
        if sxs_format not in sxs_formats:
            invalid(
                f'\nThe `sxs_format` found in JSON file is "{sxs_format}"; it should be one of\n' f"    {sxs_formats}."
            )

        # Make sure the expected H5 file size matches the observed value
        json_h5_file_size = json_data.get("validation", {}).get("h5_file_size", 0)
        if json_h5_file_size != h5_size:
            invalid(
                f"\nMismatch between `validation/h5_file_size` key in JSON file ({json_h5_file_size}) "
                f'and observed file size ({h5_size}) of "{h5_path}".'
            )

        # Make sure the expected H5 file hash matches the observed value
        if check_md5:
            md5sum = md5checksum(h5_path)
            json_md5sum = json_data.get("validation", {}).get("md5sum", "")
            if json_md5sum != md5sum:
                invalid(f"\nMismatch between `validation/md5sum` key in JSON file and observed MD5 checksum.")

    with h5py.File(h5_path, "r") as f:
        # Make sure this is our format
        sxs_format = f.attrs["sxs_format"]
        if sxs_format not in sxs_formats:
            raise ValueError(
                f'The `sxs_format` found in H5 file is "{sxs_format}"; it should be one of\n' f"    {sxs_formats}."
            )

        # Ensure that the 'validation' keys from the JSON file are the same as in this file
        n_times = f.attrs["n_times"]
        json_n_times = json_data.get("validation", {}).get("n_times", 0)
        if json_n_times != n_times:
            invalid(
                f"\nNumber of time steps in H5 file ({n_times}) "
                f"does not match expected value from JSON ({json_n_times})."
            )

        # Read the raw data
        sizeof_float = 8
        sizeof_complex = 2 * sizeof_float
        ell_min = f.attrs["ell_min"]
        ell_max = f.attrs["ell_max"]
        shuffle_widths = tuple(f.attrs["shuffle_widths"])
        unshuffle = scri.utilities.multishuffle(shuffle_widths, forward=False)
        n_modes = ell_max * (ell_max + 2) - ell_min ** 2 + 1
        i1 = n_times * sizeof_float
        i2 = i1 + n_times * sizeof_complex * n_modes
        uncompressed_data = bz2.decompress(f["data"][...])
        t = np.frombuffer(uncompressed_data[:i1], dtype=np.uint64)
        data = np.frombuffer(uncompressed_data[i1:i2], dtype=np.uint64)
        log_frame = np.frombuffer(uncompressed_data[i2:], dtype=np.uint64)

    # Unshuffle the raw data
    t = unshuffle(t)
    data = unshuffle(data)
    log_frame = unshuffle(log_frame)

    # Reshape and re-interpret the data
    t = t.view(np.float64)
    data = data.reshape((-1, n_times)).T.copy().view(complex)
    log_frame = log_frame.reshape((-1, n_times)).T.copy().view(np.float64)

    # Un-XOR the data
    t = unxor(t)
    data = unxor(data)
    log_frame = unxor(log_frame)

    frame = np.exp(quaternion.as_quat_array(np.insert(log_frame, 0, 0.0, axis=1)))

    w = scri.WaveformModes(
        t=t,
        frame=frame,
        data=data,
        frameType=scri.Corotating,
        dataType=dataType,
        m_is_scaled_out=True,
        r_is_scaled_out=True,
        ell_min=ell_min,
        ell_max=ell_max,
    )
    w.convert_from_conjugate_pairs()
    w.json_data = json_data

    return w, log_frame
