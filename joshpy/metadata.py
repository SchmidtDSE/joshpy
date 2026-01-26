
"""Structures describing simulation metadata parsed from Josh code.

License: BSD-3-Clause
"""

import typing

OPT_FLOAT = typing.Optional[float]


class SimulationMetadata:
  """Information about metadata from a simulation including grid initialization information."""

  def __init__(self, start_x: float, start_y: float, end_x: float, end_y: float, 
               patch_size: float, min_longitude: OPT_FLOAT = None, min_latitude: OPT_FLOAT = None,
               max_longitude: OPT_FLOAT = None, max_latitude: OPT_FLOAT = None):
    """Create a new metadata record.
    
    Args:
      start_x: The minimum horizontal position of a patch in grid space where coordinates in 
        degrees are automatically converted to a grid with 0,0 in upper left.
      start_y: The minimum vertical position of a patch in grid space where coordinates in 
        degrees are automatically converted to a grid with 0,0 in upper left.
      end_x: The maximum horizontal position of a patch in grid space.
      end_y: The maximum vertical position of a patch in grid space.
      patch_size: The size of each patch or cell, typically 1.
      min_longitude: The minimum longitude within this grid. Defaults to None.
      min_latitude: The minimum latitude within this grid. Defaults to None.
      max_longitude: The maximum longitude within this grid. Defaults to None.
      max_latitude: The maximum latitude within this grid. Defaults to None.
    """
    self._start_x = start_x
    self._start_y = start_y
    self._end_x = end_x
    self._end_y = end_y
    self._patch_size = patch_size
    self._min_longitude = min_longitude
    self._min_latitude = min_latitude
    self._max_longitude = max_longitude
    self._max_latitude = max_latitude

  def get_start_x(self) -> float:
    """Gets the minimum horizontal position of a patch in grid space.
    
    Returns:
      float: The starting X coordinate where coordinates in degrees are automatically converted to a 
        grid with 0,0 in upper left.
    """
    return self._start_x

  def get_start_y(self) -> float:
    """Gets the minimum vertical position of a patch in grid space.
    
    Returns:
      float: The starting Y coordinate where coordinates in degrees are automatically converted to a 
        grid with 0,0 in upper left.
    """
    return self._start_y

  def get_end_x(self) -> float:
    """Gets the maximum horizontal position of a patch in grid space.
    
    Returns:
      float: The ending X coordinate in grid space.
    """
    return self._end_x

  def get_end_y(self) -> float:
    """Gets the maximum vertical position of a patch in grid space.
    
    Returns:
      float: The ending Y coordinate in grid space.
    """
    return self._end_y

  def get_patch_size(self) -> float:
    """Gets the size of each patch/cell in the grid.
    
    Returns:
      float: The patch size, typically 1.
    """
    return self._patch_size

  def get_min_longitude(self) -> OPT_FLOAT:
    """Gets the minimum longitude within this grid.
    
    Returns:
      float: The minimum longitude, or None if grid not defined in degrees.
    """
    return self._min_longitude

  def get_min_latitude(self) -> OPT_FLOAT:
    """Gets the minimum latitude within this grid.
    
    Returns:
      float: The minimum latitude, or None if grid not defined in degrees.
    """
    return self._min_latitude

  def get_max_longitude(self) -> OPT_FLOAT:
    """Gets the maximum longitude within this grid.
    
    Returns:
      float: The maximum longitude, or None if grid not defined in degrees.
    """
    return self._max_longitude

  def get_max_latitude(self) -> OPT_FLOAT:
    """Gets the maximum latitude within this grid.
    
    Returns:
      float: The maximum latitude, or None if grid not defined in degrees.
    """
    return self._max_latitude

  def has_degrees(self) -> bool:
    """Determine if this record has latitude and longitude specified.
    
    Returns:
      bool: True if latitude and longitudes are specified and false otherwise.
    """
    has_longitude = self.get_min_longitude() is not None and self.get_max_longitude() is not None
    has_latitude = self.get_min_latitude() is not None and self.get_max_latitude() is not None
    return has_longitude and has_latitude
