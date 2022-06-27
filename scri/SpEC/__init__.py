# Copyright (c) 2021, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>
"""Submodule for operating on SpEC waveform files"""

from .. import __version__

from .com_motion import com_motion, estimate_avg_com_motion, remove_avg_com_motion
from .file_io import read_from_h5, write_to_h5, create_abd_from_h5, rpxmb
