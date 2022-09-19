************************************
Tutorial: WaveformModes/WaveformGrid
************************************

In addition to :meth:`scri.asymptotic_bondi_data.AsymptoticBondiData`, there
are two classes that can be used together to work with waveform data:
:meth:`scri.WaveformModes` and :meth:`scri.WaveformGrid`.  Eventually, these
will be deprecated by the AsymptoticBondiData class, but at the current time
there are still several features only available to ``WaveformModes``
(``WaveformGrid``).  A ``WaveformModes`` (``WaveformGrid``) holds the data for
one waveform, and lacks the features for complicated expressions involving
multiple waveform types.  For tasks of that sort, ``AsymptoticBondiData``
should be used instead.

There are two ways to represent waveform data.  The most straightfoward way is
to compute the values of the field on a grid of points over the sphere.  The
``WaveformGrid`` object is used for this purpose.  More optimally, however, the
data can be expanded in a basis of spin-weighted spherical harmonics (SWSHs)
and the coefficients of each mode can be stored.  The ``WaveformModes`` object
is used to store and analyze waveform data stored as coefficients of SWSH
modes.

===============================
Creating a WaveformModes Object
===============================

Let's say we have a set of SWSH coefficients representing waveform data for
gravitational wave strain, ``my_strain_data``.  Since the strain has
spin-weight :math:`s=-2`, any coefficients for modes with :math:`\ell < |s|`
will be zero.  ``WaveformModes`` allows us to store the the data starting with
:math:`\ell = 2` and ignore the lower modes that would all be zero.  We can
create such a ``WaveformModes`` object as follows:

.. testsetup::

  import numpy as np
  import scri
  my_strain_data = np.zeros((100,77), dtype=complex)

.. doctest::

  >>> h = scri.WaveformModes(
  ...       dataType = scri.h,
  ...       t = np.linspace(0, 10, 100),
  ...       data = my_strain_data,
  ...       ell_min = 2,
  ...       ell_max = 8,
  ...       frameType = scri.Inertial,
  ...       r_is_scaled_out = True,
  ...       m_is_scaled_out = True,
  ... )

Here we have assumed that the leading fall-off behavior of the data has been
scaled out (i.e., ``r_is_scaled_out = True``) and that the data has been scaled
by the total initial mass of the system (i.e., ``m_is_scaled_out = True``).
See the documentation on :meth:`scri.WaveformModes` for complete details about
each option.  We have also set the dataType to be that of gravitational wave
strain ``scri.h``.  The avialable dataTypes are: ``scri.UnknownDataType``,
``scri.psi0``, ``scri.psi1``, ``scri.psi2``, ``scri.psi3``, ``scri.psi4``,
``scri.h``, the time derivative of the strain ``scri.hdot``, the Newman-Penrose
shear ``scri.sigma``, and ``scri.news``.

A ``WaveformGrid`` object can be created in the same way.  For example, let's
say we have an grid of 144 equidistant points on the sphere at which the strain
has been evaluated, and we have an array ``my_strain_grid_data`` of this strain
data.  We can create the following ``WaveformGrid``:

.. testsetup::

  my_strain_grid_data = np.zeros((100,144), dtype=complex)

.. doctest::

  >>> h = scri.WaveformGrid(
  ...       dataType = scri.h,
  ...       t = np.linspace(0, 10, 100),
  ...       data = my_strain_grid_data,
  ...       n_theta = 12,
  ...       n_phi   = 12,
  ...       frameType = scri.Inertial,
  ...       r_is_scaled_out = True,
  ...       m_is_scaled_out = True,
  ... )

The points in the array correspond to the following points on the sphere:

.. code-block:: python

  >>> grid_points = np.array([
  >>>     (theta, phi)
  >>>     for theta in np.linspace(0.0, np.pi, 2*h.ell_max+1, endpoint=True)
  >>>     for phi in np.linspace(0.0, 2*np.pi, 2*h.ell_max+1, endpoint=False)
  >>> ])

------------------------------
Loading a WaveformModes Object
------------------------------

Depending on the format of the waveform in the HDF5 file, there are several
ways to load the data directly into a `WaveformModes` object:

.. code-block:: python

  >>> # For waveforms from the SXS Catalog:
  >>> h = scri.SpEC.read_from_h5("path/to/rhOverM_Asymptotic_GeometricUnits_CoM.h5/Extrapolated_N4.dir")

  >>> # For waveforms extrapolated by scri:
  >>> h = scri.SpEC.read_from_h5("path/to/rhOverM_Extrapolated_N4.h5")

  >>> # For RPXMB-compressed waveforms:
  >>> h = scri.rpxmb.load(
  ...       "path/to/rhOverM_Extrapolated_N4_RPXMB.h5"
  ... )[0].to_inertial_frame()

More information needs to be passed into ``read_from_h5`` when trying to load a
finite-radius file.  For example, if we are loading a strain waveform with data
beloning to extraction radius :math:`R = 123\, M`.  Then we would need to do
the following:

.. code-block:: python

  >>> h = scri.SpEC.read_from_h5(
  ...       "path/to/rh_FiniteRadii_CodeUnits.h5/R0123.dir",
  ...       dataType = scri.h,
  ...       frameType = scri.Inertial,
  ...       r_is_scaled_out = True,
  ...       m_is_scaled_out = True,
  ... )

In addition to this, there are several templates for generating sample
waveforms that can be loaded quickly and easily.  See the documentation on
:meth:`scri.sample_waveforms` for all the options available.  For example, a
post-Newtonian waveform can be quickly generated by using the
:meth:`scri.sample_waveforms.fake_precessing_waveform` function:

.. doctest::

  >>> h = scri.sample_waveforms.fake_precessing_waveform(
  ...       t_0 = 0.0,
  ...       t_1 = 1000.0,
  ...       dt  = 0.1,
  ...       ell_max = 4,
  ...       mass_ratio = 1.0,
  ...       precession_opening_angle = 0.0,
  ... )

==========================
Working with WaveformModes
==========================

If we have a ``WaveformModes`` object named ``h``, the time array of the
waveform can be accessed by calling ``h.t`` and the data array can be accessed
by calling ``h.data``.  Individual modes can be accessed by the ``h.index``
function.  Alternatively, you can use the ``spherical_functions.LM_index``
function from the `spherical_functions
<https://github.com/moble/spherical_functions>`_ module.  This can be aliased
to ``lm`` for convenience, as done below:

.. code-block:: python

  >>> # Get the (2,1) mode of h
  >>> l, m = 2, 1
  >>> h.data[:, h.index(l,m,h.ell_min)]

  >>> # Alternatively:
  >>> from spherical_functions import LM_index as lm
  >>> h.data[:, lm(l,m,h.ell_min)]

We can convert between ``WaveformModes`` and ``WaveformGrid``:

.. code-block:: python

  >>> # Convert from WaveformModes to WaveformGrid:
  >>> h_grid = h_modes.to_grid();

  >>> # Convert from WaveformGrid to WaveformModes:
  >>> h_modes = h_grid.to_modes()

  >>> # You can also reduce the number of modes when converting to WaveformModes:
  >>> new_lmax = 5
  >>> h_modes = h_grid.to_modes(new_lmax)

There are many built-in functions that can be performed with ``WaveformModes``.
See the documentation of :meth:`scri.WaveformModes` for the complete details,
but to name a few of the functions:

.. code-block:: python

  >>> # To interpolate the data onto a new time array:
  >>> h.interpolate(new_t)

  >>> # Returns the data array with a derivative with respect to h.t
  >>> h.data_dot

  >>> # Returns the data array with a second derivative with respect to h.t
  >>> h.data_ddot

  >>> # Returns the data array with an anti-derivative with respect to h.t
  >>> h.data_int

  >>> # Returns the data array with a second anti-derivative with respect to h.t
  >>> h.data_iint

  >>> # For a WaveformModes object of data type scri.h ONLY, we can
  >>> # compute the following fluxes:
  >>> h.energy_flux
  >>> h.angular_momentum_flux
  >>> h.momentum_flux

  >>> # To apply a series of eth and/or ethbar derivatives:
  >>> h.apply_eth('++--', eth_convention='GHP')

See the documentation on :meth:`scri.WaveformModes.apply_eth` for details on
applying the :math:`\eth` and :math:`\bar{\eth}` operators.

===================
BMS Transformations
===================

Boosts, spacetime translations, supertranslations, and a simple frame rotation
can all be performed with the :meth:`scri.WaveformModes.transform` function.
function.  See the documentation of that function (and the underlying function
:meth:`scri.WaveformGrid.from_modes`) for details.  The transformation is not
performed in place, so it will return a new object with the transformed data:

.. doctest::

  >>> h_prime = h.transform(
  ...     space_translation=[-1., 4., 0.2],
  ...     boost_velocity=[0., 0., 1e-2],
  ... )

The ABD class also supports more advanced frame rotations that interfaces with
the `quaternion <https://github.com/moble/quaternion>`_ python module.  Given a
unit quaternion ``R`` or an array of unit quaterions ``R``, you can perform a
rotation of the data:

.. testsetup::

  import quaternion
  R = quaternion.one

.. doctest::

  >>> rotated_h = h.rotate_decomposition_basis(R)

  >>> # Also:
  >>> rotated_h = h.rotate_physical_system(R)

This will return the rotated quantity, which will also store the rotor or array
of rotors that made the transformation.

There are functions to go to the corotating or coprecessing frame too.  At any
point you can undo all the frame rotations by going back to the inertial frame:

.. code-block:: python

  >>> h.to_corotating_frame()
  >>> h.to_coprecessing_frame()
  >>> h.to_inertial_frame()

The quaternion or array of quaternions that define the frame can be accessed by:

.. code-block:: python

  >>> h.frame

These components of a BMS transformation can also all be stored in the
:meth:`scri.bms_transformations.BMSTransformation` class, which allows for things like
reording the components of the BMS transformation, inverting BMS transformations, and
composing BMS transformations. For more, see :ref:`bms_transformations`.
