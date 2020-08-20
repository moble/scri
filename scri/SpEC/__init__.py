# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>
"""Submodule for operating on SpEC waveform files"""

from .._version import __version__

from .com_motion import com_motion, estimate_avg_com_motion, remove_avg_com_motion
from .file_io import read_from_h5, write_to_h5

from .memory import (
    mem_mass_aspect_contribution,
    mem_energy_flux_contribution,
    mem_angular_momentum_aspect_contribution,
    mem_angular_momentum_flux_contribution,
    add_memory,
    BMS_strain,
)
