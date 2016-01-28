#!/usr/bin/env python

# Copyright (c) 2015, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/scri/blob/master/LICENSE>

import distutils.core
from auto_version import calculate_version, build_py_copy_version


distutils.core.setup(name='scri',
                     version=calculate_version(),
                     description='Manipulating time-dependent functions of spin-weighted spherical harmonics',
                     author='Michael Boyle',
                     # author_email='',
                     url='https://github.com/moble/scri',
                     package_dir={'scri': '.'},
                     packages=['scri', 'scri.pn', 'scri.SpEC'],
                     requires=['numpy', 'scipy', 'quaternion', 'spherical_functions'],
                     cmdclass={'build_py': build_py_copy_version},
                     )
