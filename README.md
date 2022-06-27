[![Test and deploy](https://github.com/moble/scri/actions/workflows/build.yml/badge.svg)](https://github.com/moble/scri/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/scri/badge/?version=latest)](https://scri.readthedocs.io/en/latest/?badge=latest)
[![PyPI Version](https://img.shields.io/pypi/v/scri?color=)](https://pypi.org/project/scri/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/scri.svg?color=)](https://anaconda.org/conda-forge/scri)
[![MIT License](https://img.shields.io/github/license/moble/scri.svg)](https://github.com/moble/scri/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.4041971.svg)](https://doi.org/10.5281/zenodo.4041971)

Scri
====

Python/numba code for manipulating time-dependent functions of spin-weighted
spherical harmonics

## Citing this code

If you use this code for academic work (I can't actually imagine any other use
for it), please cite the latest version that you used in your publication. The DOI is:

* DOI: [10.5281/zenodo.4041972](https://doi.org/10.5281/zenodo.4041972) ([BibTeX entry on Zenodo](https://zenodo.org/record/4041972/export/hx#.YFNpLe1KiV4))

Also please cite the papers for/by which it was produced:

* "Angular velocity of gravitational radiation from precessing binaries and the
  corotating frame", Boyle,
  [Phys. Rev. D, 87, 104006](http://link.aps.org/doi/10.1103/PhysRevD.87.104006)
  (2013).
* "Gravitational-wave modes from precessing black-hole binaries", Boyle *et
  al.*, http://arxiv.org/abs/1409.4431 (2014).
* "Transformations of asymptotic gravitational-wave data", Boyle,
  [Phys. Rev. D, 93, 084031](http://link.aps.org/doi/10.1103/PhysRevD.93.084031)
  (2015).

Bibtex entries for these articles can be found
[here](https://raw.githubusercontent.com/moble/scri/master/scri.bib).  It might
also be nice of you to provide a link directly to this source code.


## Quick start

Assuming you have the [`anaconda`](http://continuum.io/downloads) distribution
of python (the preferred distribution for scientific applications),
installation is as simple as

```sh
conda update -y --all
conda install -c conda-forge scri
```

If you need to install `anaconda` first, it's very easy and doesn't require root permissions.  Just [download](http://continuum.io/downloads) and follow the instructions — particularly setting your `PATH`.  Also, make sure `PYTHONPATH` and `PYTHONHOME` are *not* set.  Ensure that it worked by running `python --version`.  It should say something about anaconda; if not, you probably forgot to set your `PATH`.  Now just run the installation command above.

Then, in python, you can check to make sure installation worked with

```python
import scri
w = scri.WaveformModes()
```

Now, `w` is an object to contain time and waveform data, as well as various
related pieces of information -- though it is trivial in this case, because we
haven't given it any data.  For more information, see the docstrings of `scri`,
`scri.WaveformModes`, etc.


## Dependencies

The dependencies should be taken care of automatically by the quick
installation instructions above.  However, if you run into problems (or if you
foolishly decide not to use anaconda to install things), it may be because you
are missing some or all of these:

  * Standard packages (come with full anaconda installation)
    * [`numpy`](http://www.numpy.org/)
    * [`scipy`](http://scipy.org/)
    * [`matplotlib`](http://matplotlib.org/)
    * [`h5py`](http://www.h5py.org/)
    * [`numba`](http://numba.pydata.org/)
  * My packages, available from anaconda.org and/or github
    * [`fftw`](https://github.com/moble/fftw) (not actually mine,
      but I maintain a copy for easy installation)
    * [`spinsfast`](https://github.com/moble/spinsfast) (not actually mine,
      but I maintain a copy with updated python features)
    * [`quaternion`](https://github.com/moble/quaternion)
    * [`spherical_functions`](https://github.com/moble/spherical_functions)

All these dependencies are installed automatically when you use the `conda`
command described above.  The `anaconda` distribution can co-exist with your
system python with no trouble -- you simply add the path to anaconda before
your system executables.  In fact, your system python probably needs to stay
crusty and old so that your system doesn't break, while you want to use a newer
version of python to actually run fancy new code like this.  This is what
`anaconda` does for you.  It installs into your home directory, so it doesn't
require root access.  It can be uninstalled easily, since it exists entirely
inside its own directory.  And updates are trivial.

### "Manual" installation

The instructions in the "Quick Start" section above should be sufficient, as
there really is no good reason not to use `anaconda`.  You will occasionally
hear people complain about it not working; these people have not installed it
correctly, and have other python-related environment variables that shouldn't
be there.  You don't want to be one of those people.

Nonetheless, it is possible to install these packages without anaconda -- in
principle.  The main hurdle to overcome is `numba`.  Maybe there are nice ways
to install `numba` without `anaconda`.  I don't know.  I don't care.  But if
you're awesome enough to do that, you're awesome enough to install all the
other dependencies without advice from me.  But in short, you can either use
the `setup.py` files as usual, or just use `pip`:

```sh
pip install git+git://github.com/moble/spinsfast
pip install git+git://github.com/moble/quaternion
pip install git+git://github.com/moble/spherical_functions
pip install git+git://github.com/moble/scri
```

And since you're just *soooo* cool, you already know that the `--user` flag is
missing from those commands because you're presumably using a virtual
environment, hotshot.

(If you're really not that cool, and aren't using `virtualenv`, you might think
you should `sudo` those commands.  But there's no need if you just use the
`--user` flag instead.  That installs packages into your user directory, which
is usually a better idea.)

Note that `spinsfast` depends (for both building and running) on `fftw`.  If
you run into build problems with `spinsfast`, it probably can't find the
header or library for `fftw`.  See the documentation of my copy of `spinsfast`
[here](https://github.com/moble/spinsfast#manual-installation) for suggestions
on solving that problem.  Of course, with `conda`, `fftw` is installed in the
right place from my channel automatically.


## Documentation

Tutorials and automatically generated API documentation are available on [Read the Docs: scri](https://scri.readthedocs.io/).

## Acknowledgments

This code is, of course, hosted on github; because it is an open-source
project, the hosting is free, and all the wonderful features of github are
available, including free wiki space and web page hosting, pull requests, a
nice interface to the git logs, etc.

Every change in this code is
[auomatically tested](https://travis-ci.org/moble/scri) on
[Travis-CI](https://travis-ci.org/).  This is a free service (for open-source
projects like this one), which integrates beautifully with github, detecting
each commit and automatically re-running the tests.  The code is downloaded and
installed fresh each time, and then tested, on both versions of python (2 and
3).  This ensures that no change I make to the code breaks either installation
or any of the features that I have written tests for.

Every change to this code is also recompiled automatically, bundled into a
`conda` package, and made available for download from
[anaconda.org](https://anaconda.org/moble/scri).  Again, because this is an
open-source project all those nice features are free.

The work of creating this code was supported in part by the Sherman Fairchild
Foundation and by NSF Grants No. PHY-1306125 and AST-1333129.
