"""Type defs in support of the Python library for Josh simulations.

License: BSD-3-Clause
"""

import typing

import joshpy.virtual_file

FlatFiles = typing.List[joshpy.virtual_file.VirtualFile]
SimulationResults = typing.List[typing.List[typing.Dict]]
