***********************
Tutorial: Extrapolation
***********************

In numerical relativity simulations, waveform quantities are extracted at
finite radii in the simulation domain.  These extraction radii are usually at
distances :math:`R\lesssim \mathcal{O}(10^3 M)`.  As a result, the waveforms
are contaminated by gauge and near-field effects.  One way to compute the
waveform at asymptotic null infinity :math:`\mathscr{I}^+` is by using an
extrapolation procedure in post-processing.  Such a procedure is provided by
``scri`` through the function :meth:`scri.extrapolation.extrapolate`; see
`arXiv:2010.15200 <https://arxiv.org/abs/2010.15200>`_ for details.

=====================================
Extrapolating Finite-Radius Waveforms
=====================================

This function will accept an HDF5 file containing the coefficients of a
waveform quantity expanded in the appropriate spin-weighted spherical harmonic
basis.  The format should be that of the ``rh_FiniteRadii_CodeUnits.h5`` files
in the `SXS Waveform Catalog
<https://data.black-holes.org/waveforms/index.html>`_, which will now be
described in detail.

The internal group structure of the HDF5 file should be as follows.  The root
group should contain one group for each extraction radius, named ``R####.dir``
where ``####`` is a four digit number corresponding to the extraction radius.
For example, if an extraction radius is :math:`123\, M`, the group should be
named ``R0123.dir``.  Within each extraction radius group, there should be
datasets named ``Y_l#_m#.dat`` corresponding to each :math:`(\ell,m) = (\#,\#)`
mode.  For example, the modes :math:`(3,2)` and :math:`(4,-2)` would be named
``Y_l3_m2.dat`` and ``Y_l4_-2.dat``.  Each dataset should be of size
``(n_times, 3)``, where ``n_times`` is the number of timesteps.  The first
column should be the simulation time, the second column should be the real part
of the coefficient, and the third column should be the imaginary part of the
coefficient.  The array of times should be the same for every mode and every
extraction radius.  Isn't this quite a waste of space? Absolutely.

To extrapolate the finite radius waveform data,

.. code-block:: python

  >>> scri.extrapolate(
  ...     InputDirectory = "path/to/FiniteRadii_waveform_dir",
  ...     OutputDirectory = "path/to/output_dir",
  ...     DataFile = "waveform_FiniteRadii_CodeUnits.h5",
  ...     ChMass = 1.0, # or whatever the initial system (Christodoulou) mass is.
  ...     UseStupidNRARFormat = True,
  ...     DifferenceFiles = '',
  ...     PlotFormat = '', 
  ... )

Despite the fact that the NRAR format for HDF5 files (described in detail
above) is quite wasteful, it's better to output the extrapolated files in this
format for two reasons.  First, the NRAR format extrapolated files interface
more widely with functions in ``scri``.  Second, these files can be highly
compressed with the RPXMB scheme described in the next section below.  The
RPXMB-compressed files are far smaller than the default scri-format
extrapolated files.

The last two options, ``DifferenceFiles=''`` and ``PlotFormat=''``, are set to
supress the output of diagnostic plots.  Only do this if you have your own set
of diagnostic tests to measure the performance of the extrapolation.  If you
wish the default diagnostic plots to be produced then leave off the last two
options:

.. code-block:: python

  >>> # Outputs diagnostic plots in addition to the waveforms
  >>> scri.extrapolate(
  ...     InputDirectory = "path/to/FiniteRadii_waveform_dir",
  ...     OutputDirectory = "path/to/output_dir",
  ...     DataFile = "waveform_FiniteRadii_CodeUnits.h5",
  ...     ChMass = 1.0, # or whatever the initial system (Christodoulou) mass is.
  ...     UseStupidNRARFormat = True,
  ... )

For more options of extrapolation function, see the documentation for
:meth:`scri.extrapolation.extrapolate`.

============================================
Compressing Extrapolated Waveforms with RPXMB
============================================

Extrapolated waveforms in the NRAR format (described in the previous section)
use an extravagant amount of needless space.  Instead, a highly compressed HDF5
file can be produced using ``scri.rpxmb``.  Files can be reduced by anywhere
from a factor 6 to a factor of 10, depending on the waveform data.

This compression can be performed in ``scri``.  For example, let's say we want
to compress an extrapolated file ``rhOverM_Extrapolated_N4.h5`` and output the
compressed waveform to the same directory with the name
``rhOverM_Extrapolated_N4_RPXMB.h5``.  This can be done as follows:

.. code-block:: python
  
  >>> w_in = scri.SpEC.read_from_h5("path/to/rhOverM_Extrapolated_N4.h5")
  >>> scri.rpxmb.save(
  ...     w_in,
  ...     "path/to/rhOverM_Extrapolated_N4_RPXMB.h5",
  ... )
