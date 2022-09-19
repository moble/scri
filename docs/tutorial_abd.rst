*****************************
Tutorial: AsymptoticBondiData
*****************************

The :meth:`scri.asymptotic_bondi_data.AsymptoticBondiData` class (or ABD for short)
is a convenient way to store and work with the standard set of asymptotic waveform
quantities: the five Weyl scalars :math:`(\Psi_4, \Psi_3, \Psi_2, \Psi_1, \Psi_0)`
and the Newman-Penrose shear :math:`\sigma`. These waveform quantities can be
expanded in a spin-weighted spherical harmonic (SWSH) basis, and the coefficients of
each mode will be stored in the ABD object.

The ABD class makes it simple to compute BMS charges and fluxes, and it serves as
a vehicle for elaborate calculations invovling asymptotic data. See the class
documentation for more details on all it has to offer. This tutorial is by no
means an exhaustive overview of the class.

======================================
Creating an AsymptoticBondiData Object
======================================

To create an ABD object, we first need to define the time series that all the
waveform quantities share, the highest value of :math:`\ell` in the SWSH expansion,
and how we will treat values of :math:`\ell` larger than ``ell_max`` during
operations:

.. testsetup::

  import numpy as np
  import scri

.. doctest::

  >>> abd = scri.asymptotic_bondi_data.AsymptoticBondiData(
  ...     time = np.linspace(0,10,11),
  ...     ell_max = 8,
  ...     multiplication_truncator = max,
  ... )

When multiplying two fields, the SWSH expansion of the product will require a higher
``ell_max``. Setting ``multiplication_truncator=sum`` will choose the ``ell_max`` of
the product of two fields to be the ``sum`` of ``ell_max`` for each field. Here we
defined each waveform quantity in ``abd`` to have ``ell_max=8``, which means that
the product :math:`\sigma\Psi_4` will be represented by ``ell_max = 16``. This means
that the total modes stored for ``ell_max = 16`` is 289 modes, as opposed to 81 modes
for ``ell_max = 8``. Although it is more correct to keep all these modes, given the
precision of numerical relativity (NR) waveforms it is often sufficient to set
``multiplication_truncator=max`` as we have done here. This will instead choose the
``ell_max`` of the product of the two fields to be the ``max`` of the values of
``ell_max`` for each field. In this case, the product :math:`\sigma\Psi_4` will
be represented by ``ell_max=8``.

Now that ``abd`` is set up, we can proceed to add some asymptotic data to it. The
data should be a ``numpy.ndarray`` of shape ``(n_times, (ell_max + 1)**2)`` , where
``n_times`` is the size of the time array passed into the definition of ``abd`` above.
This array must contain the coefficients of the SWSH modes starting from :math:`\ell=0`,
even these modes are all zeros. The :math:`(\ell,m)` modes must also be in the following order:

(0,0), (1,-1), (1,0), (1,1), (2,-2), (2,1), (2,0),...

If we have an array ``psi4_mode_data`` of coefficients
for :math:`\Psi_4`, then we can load this into ``abd`` by:

.. testsetup::

  psi4_mode_data = np.zeros((11,81))

.. doctest::

  >>> abd.psi4 = psi4_mode_data

You can load the data for ``abd.sigma``, ``abd.psi3``, ``abd.psi2``, ``abd.psi1``,
and ``abd.psi0`` in the same way. There are two crucial points to note for numerical
relativists. First, the ABD class uses the Moreschi-Boyle convention, which is *not*
the standard convention used in NR. See Appendices B and C of
`arXiv:2010.15200 <https://arxiv.org/abs/2010.15200>`_ for more information. Second,
the Newman-Penrose shear :math:`\sigma` is *not* the gravitational-wave strain :math:`h`.
In the Moreschi-Boyle convention, we have the relation :math:`h = \bar{\sigma}`, but
this relation is not gauranteed in every convention and only holds for asymptotic
quantities.

To make the ABD interface more nicely with NR waveforms, there is a tool for directly
creating an ABD object from the HDF5 files of the
`SXS waveform catalog <https://data.black-holes.org/waveforms/index.html>`_ or any
other catalog using the same file format:

.. code-block:: python

  >>> abd = scri.SpEC.create_abd_from_h5(
  ...     "SXS",
  ...     h    = "/path/to/rhOverM_file.h5",
  ...     Psi4 = "/path/to/rMPsi4_file.h5",
  ...     Psi3 = "/path/to/r2Psi3_file.h5",
  ...     Psi2 = "/path/to/r3Psi2OverM_file.h5",
  ...     Psi1 = "/path/to/r4Psi1OverM2_file.h5",
  ...     Psi0 = "/path/to/r5Psi0OverM3_file.h5",
  ... )

The first argument specifies three different ways that the HDF5 files can be
internally structured. We currently support ``SXS`` for extrapolated asymptotic
files in the the SXS catalog, ``CCE`` for asymptotic files produced by
`SpECTRE CCE <https://spectre-code.org/index.html>`_, and ``RPXMB`` for compressed
waveform files. See the documentation for :meth:`scri.SpEC.file_io.create_abd_from_h5`
for more information about these formats. This loader will convert the data into
the Moreschi-Boyle convention and set :math:`\sigma = \bar{h}` for ``abd.sigma``.

=================================
Calculations with ModesTimeSeries
=================================

With a fully loaded ABD in hand, let's do some computations!
The time array can be accessed by:

.. code-block:: python

  >>> abd.t   # or abd.u

and the data for any individual quantity can be accessed by:

.. code-block:: python

  >>> abd.sigma
  >>> abd.psi4
  >>> abd.psi3
  >>> abd.psi2
  >>> abd.psi1
  >>> abd.psi0

Individual modes of ``abd.psi4`` (for example) can be accessed by the ``abd.psi4.index``
function. Alternatively, you can use the ``spherical_functions.LM_index`` function from
the `spherical_functions <https://github.com/moble/spherical_functions>`_ module. This
can be aliased to ``lm`` for convenience, as done below. The third argument
of ``LM_index`` is ``ell_min``, but we always have ``ell_min=0`` for ABD.

.. code-block:: python

  >>> # Get the (2,1) mode of Psi4
  >>> l, m = 2, 1
  >>> abd.psi4[:, abd.psi4.index(l,m)]

  >>> # Alternatively:
  >>> from spherical_functions import LM_index as lm
  >>> abd.psi4[:, lm(l,m,0)]

The data for each quantity is stored as a :meth:`scri.modes_time_series.ModesTimeSeries`:

.. doctest::

  >>> type(abd.sigma)
  <class 'scri.modes_time_series.ModesTimeSeries'>

There are many built-in functions that can be performed with a ``ModesTimeSeries``, and
multiple operations can be composed easily. Multiple operations will be performed in order
from left to right.

.. code-block:: python

  >>> # take a derivative or two
  >>> psi4_dot  = abd.psi4.dot
  >>> psi4_ddot = abd.psi4.ddot

  >>> # Integrate once or twice
  >>> psi4_int  = abd.psi4.int
  >>> psi4_iint = abd.psi4.iint

  >>> # Apply the eth or ethbar operator (using the GHP definition,
  >>> # which is the one that is natural to the Moreschi-Boyle convention)
  >>> eth_psi4    = abd.psi4.eth_GHP
  >>> ethbar_psi4 = abd.psi4.ethbar_GHP

  >>> # If you really want to use the Newman-Penrose eth operators then you can do:
  >>> eth_psi4    = abd.psi4.eth
  >>> ethbar_psi4 = abd.psi4.ethbar

  >>> # Get fancy and combine them together
  >>> abd.sigma.dot.eth_GHP.eth_GHP

These operations will respect and update the spin-weight accordingly:

.. doctest::

  >>> # Check the spin-weight
  >>> abd.sigma.s
  2
  >>> abd.sigma.bar.s
  -2
  >>> abd.sigma.dot.eth_GHP.eth_GHP.s
  4

We can add ABD quantities together, but only in a way that makes sense. An error will
be thrown for adding quantities of different spin weights:

.. code-block:: python

  >>> # This will throw an error
  >>> abd.psi4 + abd.psi3

  >>> # This will work!
  >>> abd.psi4.eth_GHP + abd.psi3

There are two ways to perform multiplication with ``ModesTimeSeries`` quantities. Here
we do not mean multiplying the mode coefficients together, but properly multiplying the
fields and return the mode coefficients of the product. The more straightforward way
to perform a multiplication is:

.. code-block:: python

  >>> sigma_psi4 = abd.sigma * abd.psi4

This will combine the mode coefficients with Wigner-3j symbols to compute the resulting
mode coefficients of the product. The benefits of this approach are that it is
straightforward to code and it does not suffer from aliasing effects. The downside is that
it take a long time to run, so make sure you store the result as a variable if you will
be using it more than once!

The second way to perform a multiplication is:

.. code-block:: python

  >>> sigma_psi4 = abd.sigma.grid_multiply(abd.psi4)

This will convert ``abd.sigma`` and ``abd.psi`` from a spectral representation to a
physical-space representation by evaluating the fields on a grid of points on
:math:`\mathscr{I}^+`. The mutliplication is performed pointwise and then transformed
back to a spectral representation. This approach is much faster. However, if the value
of ``ell_max`` is not high enough then aliasing effects might arise. See the documentation
on :meth:`scri.modes_time_series.ModesTimeSeries.grid_multiply` for options to adjust
``ell_max`` during the operation.

This just scratches the surface of all you can do with the ABD class. See the
documentation on :meth:`scri.asymptotic_bondi_data.AsymptoticBondiData` and
:meth:`scri.modes_time_series.ModesTimeSeries` to explore more functions.

------
Caveat
------

When taking the real part, imaginary part, or the complex conjugate, be very careful
to know whether you are acting on the quantity as a field or just the modes of the
quantity. For example:

.. code-block:: python

  >>> # This takes the real part of the quantity Psi2, and then
  >>> # returns the modes of Re(Psi2). The mode weights are still complex!
  >>> abd.psi2.real

  >>> # This returns the real part of the mode weights of Psi2.
  >>> # This is an array of real numbers!
  >>> abd.psi2.ndarray.real

  >>> # This returns the modes of the complex conjugate of sigma.
  >>> # This is usually what you want.
  >>> abd.sigma.bar

  >>> # This returns the complex conjguate of the modes of sigma.
  >>> # This is usually NOT what you want.
  >>> np.conjugate(abd.sigma.ndarray)

===================
BMS Transformations
===================

Spacetime translations, supertranslations, frame rotations, and boosts can all be performed
with the :meth:`scri.asymptotic_bondi_data.AsymptoticBondiData.transform` function.
See the documentation of the function for details. All the quantities in
the ABD object will be transformed together. The transformation is not performed
in place, so it will return a new ABD with the transformed data:

.. doctest::

  >>> abd_prime = abd.transform(
  ...     space_translation=[-1., 4., 0.2],
  ...     boost_velocity=[0., 0., 1e-2],
  ... )

These components of a BMS transformation can also all be stored in the
:meth:`scri.bms_transformations.BMSTransformation` class, which allows for things like
reording the components of the BMS transformation, inverting BMS transformations, and
composing BMS transformations. For more, see :ref:`bms_transformations`. 
