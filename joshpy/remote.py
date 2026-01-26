"""Implementation of JoshBackend for working with a remote Josh sever.

License: BSD-3-Clause
"""

import typing

import requests

import joshpy.definitions
import joshpy.metadata
import joshpy.parse
import joshpy.strategy

PUBLIC_DEFAULT_ENDPOINT = 'https://josh-executor-prod-1007495489273.us-west1.run.app'


class ParseResult:
  """Result of invoking the parse endpoint."""

  def __init__(self, error: typing.Optional[str], simulation_names: typing.List[str],
      metadata: typing.Optional[joshpy.metadata.SimulationMetadata]):
    """Create a new record of a parse result.

    Args:
      error: The error if one encountered in interpretation or None if no error encountered.
      simulation_names: List of simulation names found in the code or empty if error encountered.
      metadata: The metadata found for the requested simulation if simulation requested and no
        errors encountered.
    """
    self._error = error
    self._simulation_names = simulation_names
    self._metadata = metadata

  def get_error(self) -> typing.Optional[str]:
    """Get the error encountered if any.

    Returns:
      The error if one encountered in interpretation or None if no error encountered.
    """
    return self._error

  def get_simulation_names(self) -> typing.List[str]:
    """Get the simulations found in the code sent to the parse endpoint.

    Returns:
      List of simulation names found in the code or empty if error encountered.
    """
    return self._simulation_names

  def get_metadata(self) -> typing.Optional[joshpy.metadata.SimulationMetadata]:
    """Get the simulation metadata found in the code.

    Returns:
      typing.Optional[joshpy.metadata.SimulationMetadata]: The metadata found for the requested
        simulation if simulation requested and no errors encountered.
    """
    return self._metadata


class RemoteJoshDecorator(joshpy.strategy.JoshBackend):
  """Implementation of JoshBackend which uses a remote Josh to run simulations."""

  def __init__(self, server: typing.Optional[str] = None, api_key: typing.Optional[str] = None):
    """Load a new copy of the WASM Josh backend.

    Args:
      server: The endpoint at which the Josh server will be found. If not provided, will use the
        public default
      api_key: The API key to use in communication with the Josh server. If not provided, will use
        an empty string.
    """
    self._server = PUBLIC_DEFAULT_ENDPOINT if server is None else server
    self._api_key = '' if api_key is None else api_key

  def get_error(self, code: str) -> typing.Optional[str]:
    """Get the error in the given Josh code if one is present.

    Args:
      code: The code to be parsed and from which error message is to be returned.

    Returns:
      The first error found described as a string or None if no errors found.
    """
    result = self._parse_simulation(code)
    return result.get_error()

  def get_simulations(self, code: str) -> typing.List[str]:
    """Get the list of simulations found within code if it interprets correctly.

    Args:
      code: The code to be parsed and from which list of simulation names is to be returned.

    Returns:
      List of simulation names found within the code.

    Raises:
      RuntimeError: Raised if the code provided has an error in it. The exception will have the
        string description of the first error.
    """
    result = self._parse_simulation(code)
    if result.get_error() is not None:
      raise RuntimeError(result.get_error())
    return result.get_simulation_names()

  def get_metadata(self, code: str, name: str) -> joshpy.metadata.SimulationMetadata:
    """Get the metadata for a specific simulation.

    Args:
      code: The code to be parsed and from which metadata should be extracted.
      name: The name of the simulation for which metadata should be returned.

    Returns:
      The metadata parsed from the given code for the given simulation.

    Raises:
      RuntimeError: Raised if the code provided has an error in it or the simulation is not found.
    """
    result = self._parse_simulation(code, name)
    if result.get_error() is not None:
      raise RuntimeError(result.get_error())
    
    metadata = result.get_metadata()
    if metadata is None:
      raise RuntimeError(f"No metadata found for simulation: {name}")
      
    return metadata

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
    endpoint = f"{self._server}/runReplicates"
    data = {
      'code': code,
      'name': name,
      'replicates': str(replicates),
      'api_key': self._api_key,
      'externalData': virtual_files
    }

    try:
      response = requests.post(endpoint, data=data, stream=True)
      if response.status_code != 200:
        raise RuntimeError(f"Server returned status code {response.status_code}")

      # Create response reader to parse streaming results
      reader = joshpy.parse.ResponseReader(lambda x: print(f"Completed {x} replicates"))
      
      # Process streaming response chunks
      for chunk in response.iter_lines(decode_unicode=True):
        if chunk:
          reader.process_response(chunk + '\n')
          
      # Get final results
      results = reader.get_complete_replicates()
      if not results:
        raise RuntimeError("No results returned from simulation")
        
      return results

    except requests.exceptions.RequestException as e:
      raise RuntimeError(f"Failed to connect to server: {str(e)}")

  def _parse_simulation(self, code: str, name: typing.Optional[str] = None) -> ParseResult:
    """Try parsing a simulation.

    Try parsing a simulation and return results from the remote. If simulation was not parsed
    successfully, returns a result with the error indicated and empty simulation names and empty
    metadata. If simulation parsed successfully, the error is indicated as None and a list of
    simulation names are provided.

    If a name is provided and parsing is successful, the metadata for the simulation of that name
    parsed and returned if the start / end is provided in degrees and the size of the  simulation is
    in meters (m, meter, meters). The other attributes are calculated through joshpy.geocode.

    Args:
      code: The code to be parsed.
      name: The name of the simulation for which metadata should be returned or None if no metadata
        should be returned.

    Returns:
      Result of parsing with error information or simulation information.
    """
    endpoint = f"{self._server}/parse"
    data = {
      'code': code,
      'api_key': self._api_key
    }
    
    if name is not None:
      data['name'] = name
      
    try:
      response = requests.post(endpoint, data=data)
      if response.status_code != 200:
        return ParseResult(
          error=f"Server returned status code {response.status_code}",
          simulation_names=[],
          metadata=None
        )
        
      # Parse response which is tab separated: status, names, metadata
      parts = response.text.split('\t')
      if len(parts) != 3:
        return ParseResult(
          error="Invalid response format from server",
          simulation_names=[],
          metadata=None
        )
        
      status, names_csv, metadata_str = parts
      
      # Check parse status
      if status != 'success':
        return ParseResult(error=status, simulation_names=[], metadata=None)
        
      # Get simulation names
      simulation_names = names_csv.split(',') if names_csv else []
      
      # Parse metadata if name was provided and metadata returned
      metadata = None
      if name is not None and metadata_str:
        try:
          metadata = self._parse_metadata(metadata_str)
        except ValueError as e:
          return ParseResult(
            error=str(e),
            simulation_names=[],
            metadata=None
          )
          
      return ParseResult(error=None, simulation_names=simulation_names, metadata=metadata)
      
    except requests.exceptions.RequestException as e:
      return ParseResult(
        error=f"Failed to connect to server: {str(e)}",
        simulation_names=[],
        metadata=None
      )

  def _parse_metadata(self, target: str) -> joshpy.metadata.SimulationMetadata:
    """Parse the string returned from the server describing the metadata for a simulation.

    Args:
      target: The string section returned from the server, specifically the third value after the
        two tab characters.

    Returns:
      joshpy.metadata.SimulationMetadata: Parsed simulation metadata.
    """
    if not target:
      raise ValueError("Empty metadata string provided")
    
    # Split into grid info parts, format is "start:end:size units"
    parts = target.split(' ', 1)
    if len(parts) != 2:
      raise ValueError(f"Invalid metadata format: {target}")
      
    info_parts = parts[0].split(':')
    if len(info_parts) != 3:
      raise ValueError(f"Invalid grid info format: {parts[0]}")
      
    start_str, end_str, size_value = info_parts
    units = parts[1]
    
    # Parse start and end coordinates 
    start = joshpy.parse.parse_start_end_string(start_str)
    end = joshpy.parse.parse_start_end_string(end_str)
    size = joshpy.parse.parse_engine_value_string(size_value + ' ' + units)
    
    # Create metadata object with grid coordinates
    return joshpy.metadata.SimulationMetadata(
      start_x=0.0,
      start_y=0.0, 
      end_x=abs(end.get_longitude().get_value() - start.get_longitude().get_value()),
      end_y=abs(end.get_latitude().get_value() - start.get_latitude().get_value()),
      patch_size=size.get_value(),
      min_longitude=min(start.get_longitude().get_value(), end.get_longitude().get_value()),
      max_longitude=max(start.get_longitude().get_value(), end.get_longitude().get_value()),
      min_latitude=min(start.get_latitude().get_value(), end.get_latitude().get_value()),
      max_latitude=max(start.get_latitude().get_value(), end.get_latitude().get_value())
    )
