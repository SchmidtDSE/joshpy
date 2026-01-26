"""Description of the strategy interface for Josh backends.

License: BSD-3-Clause
"""

import typing

import joshpy.definitions


class JoshBackend:
  """Interface description for strategies which can perform operations on Josh code.

  Interface description for strategies which can perform operations on Josh code, either by running
  local code or executing requests to a remote backend.
  """

  def get_error(self, code: str) -> typing.Optional[str]:
    """Get the error in the given Josh code if one is present.

    Attempt to interpret the given Josh code and the first error found if one is present. Otherwise, 
    return None.

    Returns:
      typing.Optional[str]: The first error found described as a string or None if no errors found.
    """
    raise NotImplementedError('Must use implementor.')
    
  def get_simulations(self, code: str) -> typing.List[str]:
    """Get the list of simulations found within code if it interprets correctly.

    Args:
      code: The code to be parsed and from which list of simulation names is to be returned.

    Returns:
      typing.List[str]: List of simulation names found within the code.

    Raises:
      RuntimeError: Raised if the code provided has an error in it. The exception will have the
        string description of the first error.
    """
    raise NotImplementedError('Must use implementor.')

  def get_metadata(self, code: str, name: str) -> joshpy.metadata.SimulationMetadata:
    """Get the metadata for a specific simulation.

    Return the metadata parsed from the given code for the given simulation including dimensions of
    the simulation grid.

    Args:
      code: The code from which metadata should be parsed.
      name: The name of the simulation for which metadata should be parsed.

    Returns:
      joshpy.metadata.SimulationMetadata: Metadata object parsed for the requested simulation.

    Raises:
      RuntimeError: Raised if the code provided has an error in it. The exception will have the
        string description of the first error.
      ValueError: Raised if the simulation of the given name cannot be found.
    """
    raise NotImplementedError('Must use implementor.')
    
  def run_simulation(self, code: str, name: str,
      virtual_files: joshpy.definitions.FlatFiles,
      replicates: int) -> joshpy.definitions.SimulationResults:
    """Run a simulation using the current Josh backend.

    Run a simulation, printing to the console when each replicate is completed. The full result set
    will be returned with position.x and position.y as well as a step attribute along with exported 
    fields.

    Args:
      code: The code to execute.
      name: The name of the simulation from the provided code to execute.
      virtual_files: List of virutal files to provide to the simulation within its sandbox.
      replicates: The number of replicates for which the simulation should run.

    Returns:
      joshpy.definitions.SimulationResults: Outer list where each element is a replicate and each
        replicate is a list containing each data point.

    Raises:
      RuntimeError: Raised if the code provided has an error in it. The exception will have the
        string description of the first error.
      ValueError: Raised if the simulation of the given name cannot be found.
    """
    raise NotImplementedError('Must use implementor.')
