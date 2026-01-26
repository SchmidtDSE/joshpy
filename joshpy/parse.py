"""Utilties for parsing certain strings returned from the engine.

License: BSD-3-Clause
"""

import typing

import joshpy.definitions


class ResponseReader:
  """Utility to parse responses from the engine."""
  
  def __init__(self, callback):
    """Create a new reader which parses external responses.
    
    Args:
      callback: Function to call when replicates are ready.
    """
    self._replicate_reducer = {}
    self._complete_replicates = [] 
    self._callback = callback
    self._buffer = ''
    self._completed_replicates = 0

  def process_response(self, text: str):
    """Parse a response into data records.
    
    Args:
      text: The text returned by the engine.
    """
    self._buffer += text
    lines = self._buffer.split('\n')
    self._buffer = lines.pop()

    for line in (x.strip() for x in lines if x.strip()):
      intermediate = self._parse_engine_response(line)
      
      if intermediate['type'] == 'datum':
        replicate_id = intermediate['replicate']
        if replicate_id not in self._replicate_reducer:
          self._replicate_reducer[replicate_id] = []
          
        parsed = {
          'target': intermediate['datum']['target'],
          'attributes': intermediate['datum']['attributes']
        }
        self._replicate_reducer[replicate_id].append(parsed)
        
      elif intermediate['type'] == 'end':
        self._completed_replicates += 1
        self._complete_replicates.append(
          self._replicate_reducer[intermediate['replicate']]
        )
        self._callback(self._completed_replicates)

  def get_complete_replicates(self) -> joshpy.definitions.SimulationResults:
    """Get a listing of all completed replicates.
    
    Returns:
      Result from each replicate as an individual element in the list.
    """
    return self._complete_replicates

  def _parse_datum(self, source: str) -> dict:
    """Parse a single data point from a transfer string without replicate prefix.
    
    Args:
      source: The internal transfer string to parse.
      
    Returns:
      Dictionary with target name and attributes.
    """
    first_pieces = source.split(':', 1)
    target = first_pieces[0]
    attributes_str = first_pieces[1] if len(first_pieces) > 1 else ""

    attributes: typing.Dict[str, typing.Union[float, int, str]] = {}
    if not attributes_str:
      return {'target': target, 'attributes': attributes}

    pairs = attributes_str.split('\t')
    for pair in pairs:
      if not pair:
        continue
      pair_pieces = pair.split('=', 1)
      if len(pair_pieces) != 2:
        continue
        
      key, value = pair_pieces
      if key and value is not None:
        try:
          # Try parsing as number if possible
          if '.' in value:
            attributes[key] = float(value)
          else:
            attributes[key] = int(value)
        except ValueError:
          attributes[key] = value

    return {'target': target, 'attributes': attributes}

  def _parse_engine_response(self, source: str) -> dict:
    """Parse a data point from a transfer string with replicate prefix.
    
    Args:
      source: The line returned from the engine.
      
    Returns:
      Dictionary with replicate number and message type.
    """
    import re
    
    end_match = re.match(r'^\[end (\d+)\]$', source)
    if end_match:
      return {
        'replicate': int(end_match.group(1)),
        'type': 'end'
      }

    match = re.match(r'^\[(\d+)\] (.+)$', source)
    if not match:
      raise ValueError('Got malformed engine response')

    replicate = int(match.group(1))
    data = self._parse_datum(match.group(2))
    
    if not data:
      raise ValueError('Got malformed engine response')

    return {
      'replicate': replicate,
      'type': 'datum',
      'datum': data
    }


class EngineValue:
  """Value returned by the engine."""

  def __init__(self, value: float, units: str):
    """Create a new engine value record.

    Args:
      value: The numeric value.
      units: The description of the units for this value like degrees.
    """
    self._value = value
    self._units = units

  def get_value(self) -> float:
    """Get the numeric portion of this engine value.

    Returns:
      The numeric value.
    """
    return self._value

  def get_units(self) -> str:
    """Get the units portion of this engine value.

    Returns:
      The description of the units for this value like degrees.
    """
    return self._units


class StartEndString:
  """Description of a start or end string."""

  def __init__(self, longitude: EngineValue, latitude: EngineValue):
    """Create a new point parsed from a start or end string.

    Args:
      longitude: The horizontal component.
      latitude: The vertical component.
    """
    self._longitude = longitude
    self._laitutde = latitude

  def get_longitude(self) -> EngineValue:
    """Get the longitude parsed from the engine-returned string.

    Returns:
      The horizontal component.
    """
    return self._longitude

  def get_latitude(self) -> EngineValue:
    """Get the latitude parsed from the engine-returned string.

    Returns:
      The vertical component.
    """
    return self._laitutde


def parse_engine_value_string(target: str) -> EngineValue:
  """Parse an EngineValue returned from the engine.
  
  Parse an EngineValue returned from the engine which is in the string of form like follows without
  quotes: "30 m".

  Args:
    target: The string to parse as an EngineValue.

  Returns:
    Parsed EngineValue.
  """
  parts = target.strip().split(' ', 1)
  if len(parts) != 2:
    raise ValueError(f"Invalid engine value string format: {target}")
  value = float(parts[0])
  units = parts[1]
  return EngineValue(value, units)


def parse_start_end_string(target: str) -> StartEndString:
  """Parse a start or an end string.

  Parse a start or end string which may be like the following without quotes:
  "36.51947777043374 degrees latitude, -118.67203360913730 degrees longitude"

  Returns:
    Parsed version of the string.
  """
  parts = target.strip().split(',')
  if len(parts) != 2:
    raise ValueError(f"Invalid start/end string format: {target}")
    
  first_parts = parts[0].strip().split(' ')
  second_parts = parts[1].strip().split(' ')
  
  if len(first_parts) < 3 or len(second_parts) < 3:
    raise ValueError(f"Invalid coordinate format in: {target}")
    
  first_is_latitude = 'latitude' in first_parts[2]
  
  if first_is_latitude:
    latitude = EngineValue(float(first_parts[0]), first_parts[1])
    longitude = EngineValue(float(second_parts[0]), second_parts[1])
  else:
    longitude = EngineValue(float(first_parts[0]), first_parts[1])
    latitude = EngineValue(float(second_parts[0]), second_parts[1])
  
  return StartEndString(longitude, latitude)
