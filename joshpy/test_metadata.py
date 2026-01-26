import unittest

import joshpy.metadata


class TestMetadata(unittest.TestCase):
    def setUp(self):
        self.start_x = 0.0
        self.start_y = 0.0
        self.end_x = 10.0
        self.end_y = 10.0
        self.patch_size = 1.0
        self.min_longitude = -122.45
        self.min_latitude = 37.73
        self.max_longitude = -118.24
        self.max_latitude = 34.05
        
        self.metadata = joshpy.metadata.SimulationMetadata(
            self.start_x, 
            self.start_y,
            self.end_x,
            self.end_y,
            self.patch_size,
            self.min_longitude,
            self.min_latitude,
            self.max_longitude,
            self.max_latitude
        )

    def test_get_start_x(self):
        self.assertEqual(self.metadata.get_start_x(), self.start_x)

    def test_get_start_y(self):
        self.assertEqual(self.metadata.get_start_y(), self.start_y)

    def test_get_end_x(self):
        self.assertEqual(self.metadata.get_end_x(), self.end_x)

    def test_get_end_y(self):
        self.assertEqual(self.metadata.get_end_y(), self.end_y)

    def test_get_patch_size(self):
        self.assertEqual(self.metadata.get_patch_size(), self.patch_size)

    def test_get_min_longitude(self):
        self.assertEqual(self.metadata.get_min_longitude(), self.min_longitude)

    def test_get_min_latitude(self):
        self.assertEqual(self.metadata.get_min_latitude(), self.min_latitude)

    def test_get_max_longitude(self):
        self.assertEqual(self.metadata.get_max_longitude(), self.max_longitude)

    def test_get_max_latitude(self):
        self.assertEqual(self.metadata.get_max_latitude(), self.max_latitude)

    def test_has_degrees_with_coordinates(self):
        self.assertTrue(self.metadata.has_degrees())

    def test_has_degrees_without_coordinates(self):
        metadata = joshpy.metadata.SimulationMetadata(
            self.start_x,
            self.start_y,
            self.end_x,
            self.end_y,
            self.patch_size
        )
        self.assertFalse(metadata.has_degrees())

    def test_has_degrees_with_partial_coordinates(self):
        metadata = joshpy.metadata.SimulationMetadata(
            self.start_x,
            self.start_y,
            self.end_x,
            self.end_y,
            self.patch_size,
            min_longitude=self.min_longitude
        )
        self.assertFalse(metadata.has_degrees())

if __name__ == '__main__':
    unittest.main()
