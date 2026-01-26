"""Utilities to convert from grid-space to Earth-space results.

import math

Utilities to convert from grid-space to Earth-space results including a Python-based implmentation
of Haversine to support these operations.

License: BSD-3-Clause
"""

import math

import joshpy.definitions
import joshpy.metadata

EARTH_RADIUS_METERS = 6371000


def add_positions(results: joshpy.definitions.SimulationResults,
      metadata: joshpy.metadata.SimulationMetadata) -> joshpy.definitions.SimulationResults:
  """Add position.longitude and position.latitude to results.
  
  Under assumption that all replicates share the same metadata, convert from grid-space coordinates
  which report in number of patches or cells to the left and to the bottom of the origin to also
  report on position.longitude and position.latitude in degrees.

  Args:
    results: The results to be modified in place in which position.longitude and position.latitude
      will be added.
    metadata: Metadata about the simulation executed including bounds to be used in geocoding.
      The dictionaries inside will be modified in place to add a position.longitude and
      position.latitude, overwritting prior values if present.

  Returns:
    joshpy.definitions.SimulationResults: The results after modification in place.
  """
  for replicate in results:
    for entity in replicate:
      if 'position.x' in entity:
        
        longitude = metadata.get_min_longitude()
        assert longitude is not None
        
        latitude = metadata.get_max_latitude()
        assert latitude is not None
        
        top_left = EarthPoint(
          longitude,
          latitude
        )
        
        # Move east by x distance
        east_point = get_at_distance_from(
          top_left, 
          float(entity['position.x']) * metadata.get_patch_size(), 
          'E'
        )
        
        # Move south from the east point by y distance
        final_point = get_at_distance_from(
          east_point,
          float(entity['position.y']) * metadata.get_patch_size(),
          'S'  
        )
        
        entity['position.longitude'] = final_point.get_longitude()
        entity['position.latitude'] = final_point.get_latitude()
            
  return results


class EarthPoint:
  """Internal reprsentation of a geographic point on Earth."""

  def __init__(self, longitude: float, latitude: float):
    """Create a new record of a position on Earth.

    Args:
      longitude: The longitude of this point in degrees.
      latitude: The latitude of this point in degrees.
    """
    self._longitude = longitude
    self._latitude = latitude
  
  def get_longitude(self) -> float:
    """Get the east / west position of this point.
    
    Returns:
      The longitude of this point in degrees where positive numbers are east and negative west.
    """
    return self._longitude
  
  def get_latitude(self) -> float:
    """Get the north / south position of this point.
    
    Returns:
      The latitude of this point in degrees where positive numbers are north and negative south.
    """
    return self._latitude


def get_distance_meters(start: EarthPoint, end: EarthPoint) -> float:
  """Get the distance in meters between two points on Earth.
  
  Args:
    start: The starting point from which distance should be measured.
    end: The ending point to which distance should be measured.

  Returns:
    Distance between start and end in meters.
  """
  angle_lat_start = math.radians(start.get_latitude())
  angle_lat_end = math.radians(end.get_latitude())
  delta_lat = math.radians(end.get_latitude() - start.get_latitude())
  delta_long = math.radians(end.get_longitude() - start.get_longitude())
  
  a = (math.sin(delta_lat / 2) * math.sin(delta_lat / 2) 
       + math.cos(angle_lat_start) * math.cos(angle_lat_end) 
       * math.sin(delta_long / 2) * math.sin(delta_long / 2))
  
  c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
  
  return EARTH_RADIUS_METERS * c


def get_at_distance_from(start: EarthPoint, distance_meters: float, direction: str) -> EarthPoint:
  """Get a new point which is some distance from a starting point.
  
  Args:
    start: The starting point from which a new point should be derived.
    distance_meters: How far in a cardinal direction the new point should be from the starting point.
    direction: The direction as a single letter string like N, S, E, W corresponding to the cardinal 
      directions.

  Returns:
    New point which is the given distance in the given direction from the starting point.
  
  Raises:
    ValueError: If direction is not one of N, S, E, or W.
  """
  lat = math.radians(start.get_latitude())
  lng = math.radians(start.get_longitude())
  
  if direction not in ['N', 'S', 'E', 'W']:
    raise ValueError("Direction must be N, S, E, or W")
    
  new_lat = lat
  new_lng = lng
  
  if direction == 'N':
    # Moving north increases latitude
    new_lat = lat + (distance_meters / EARTH_RADIUS_METERS)
  elif direction == 'S':
    # Moving south decreases latitude  
    new_lat = lat - (distance_meters / EARTH_RADIUS_METERS)
  elif direction == 'E':
    # Moving east increases longitude, adjusted for latitude
    new_lng = lng + (distance_meters / (EARTH_RADIUS_METERS * math.cos(lat)))
  else:  # direction == 'W'
    # Moving west decreases longitude, adjusted for latitude
    new_lng = lng - (distance_meters / (EARTH_RADIUS_METERS * math.cos(lat)))
    
  # Convert back to degrees
  new_lat_degrees = math.degrees(new_lat)
  new_lng_degrees = math.degrees(new_lng)
  
  return EarthPoint(new_lng_degrees, new_lat_degrees)
