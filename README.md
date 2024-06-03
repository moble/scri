[![Test and deploy](https://github.com/moble/scri/actions/workflows/build.yml/badge.svg)](https://github.com/moble/scri/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/scri/badge/?version=latest)](https://scri.readthedocs.io/en/latest/?badge=latest)
[![PyPI Version](https://img.shields.io/pypi/v/scri?color=)](https://pypi.org/project/scri/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/scri.svg?color=)](https://anaconda.org/conda-forge/scri)
[![MIT License](https://img.shields.io/github/license/moble/scri.svg)](https://github.com/moble/scri/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.4041971.svg)](https://doi.org/10.5281/zenodo.4041971)

Scri
====

Python/numba code for manipulating time-dependent functions of spin-weighted
spherical harmonics on future null infinity

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
[here](https://raw.githubusercontent.com/moble/scri/main/scri.bib).  It might
also be nice of you to provide a link directly to this source code.


## Quick start

Note that installation is not possible on Windows due to missing FFTW support.

Installation is as simple as
```sh
conda install -c conda-forge scri
```
or
```sh
python -m pip install scri
```
If the latter command complains about permissions, you're probably using your
operating system's version of `python`, which can cause serious conflicts with
essential OS functions.  To avoid these issues,
[install conda/mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).
This will create a separate copy of python inside your home directory (avoiding
issues with permissions) which you can update independently of your OS.

Then, in python, you can check to make sure installation worked with
```python
import scri
w = scri.WaveformModes()
```
Note that scri can take a few seconds to import the first time as it compiles
some code automatically.  Here, `w` is an object to contain time and waveform
data, as well as various related pieces of information -- though it is trivial
in this case, because we haven't given it any data.  For more information, see
the docstrings of `scri`, `scri.WaveformModes`, etc.


## Documentation

Tutorials and automatically generated API documentation are available on [Read the Docs: scri](https://scri.readthedocs.io/).


## Acknowledgments

Every change to this code is recompiled automatically, bundled into a
`conda` package, and made available for download from
[anaconda.org](https://anaconda.org/moble/scri).

The work of creating this code was supported in part by the Sherman Fairchild
Foundation and by NSF Grants No. PHY-1306125 and AST-1333129.
