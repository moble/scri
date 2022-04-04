# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import sxs
from ... import WaveformModes


default_shuffle_widths = sxs.utilities.default_shuffle_widths


def save(w, file_name=None, file_write_mode="w", L2norm_fractional_tolerance=1e-10, log_frame=None, shuffle_widths=default_shuffle_widths):
    """Save a waveform in RPXMB format

    This function is essentially a backwards-compatibility wrapper for the the
    `sxs.rpxmb.save` function; see that function's docstring for details.

    """
    return sxs.rpxmb.save(
        w.to_sxs,
        file_name=file_name,
        file_write_mode=file_write_mode,
        L2norm_fractional_tolerance=L2norm_fractional_tolerance,
        log_frame=log_frame,
        shuffle_widths=shuffle_widths
    )


def load(file_name, ignore_validation=False, check_md5=True, transform_to_inertial=False, **kwargs):
    """Load a waveform in RPXMB format

    This function is essentially a backwards-compatibility wrapper for the the
    `sxs.rpxmb.load` function; see that function's docstring for full details.

    However, note that this function has slightly different behavior (for backwards
    compatibility):

      1. The default value of `ignore_validation` is `False`, which means that a
         `ValueError` is raised whenever validation fails, rather than just a
         warning.

      2. The default value of `transform_to_inertial` is `False`, which means that
         the returned waveform will still be in the corotating frame (specified
         most precisely by the `log_frame` attribute of the returned waveform).

      3. If `transform_to_inertial` is `False`, the return value is a tuple of `(w,
         log_frame)`; otherwise, it is just the waveform `w`.  In both cases,
         `w.log_frame` will contain this data.

    """
    w_sxs = sxs.rpxmb.load(
        file_name,
        ignore_validation=ignore_validation,
        check_md5=check_md5,
        transform_to_inertial=transform_to_inertial,
        **kwargs
    )
    w = WaveformModes.from_sxs(w_sxs)
    w.json_data = w_sxs.json_data
    w.log_frame = w_sxs.log_frame

    if transform_to_inertial:
        return w
    else:
        return w, w.log_frame
