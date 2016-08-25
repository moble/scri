#!/usr/bin/env python

# Copyright (c) 2016, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/quaternion/blob/master/LICENSE>

from __future__ import print_function, absolute_import

import fileinput
from auto_version import calculate_version

version_string = calculate_version().replace('.dirty', '').replace('+dirty', '')

f = fileinput.FileInput('conda/meta.yaml', inplace=True)
for line in f:
    print(line.replace("version: '1.0'", "version: '{0}'".format(version_string)), end='')
f.close()
