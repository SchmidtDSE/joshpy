import unittest

import joshpy.parse


class TestParse(unittest.TestCase):
    def test_parse_engine_value_string(self):
        test_string = "30 m"
        result = joshpy.parse.parse_engine_value_string(test_string)
        self.assertEqual(result.get_value(), 30.0)
        self.assertEqual(result.get_units(), "m")

    def test_parse_start_end_string_latitude_first(self):
        test_string = "36.51947777043374 degrees latitude, -118.67203360913730 degrees longitude"
        result = joshpy.parse.parse_start_end_string(test_string)
        self.assertEqual(result.get_longitude().get_value(), -118.67203360913730)
        self.assertEqual(result.get_longitude().get_units(), "degrees")
        self.assertEqual(result.get_latitude().get_value(), 36.51947777043374)
        self.assertEqual(result.get_latitude().get_units(), "degrees")

    def test_parse_start_end_string_longitude_first(self):
        test_string = "-118.67203360913730 degrees longitude, 36.51947777043374 degrees latitude"
        result = joshpy.parse.parse_start_end_string(test_string)
        self.assertEqual(result.get_longitude().get_value(), -118.67203360913730)
        self.assertEqual(result.get_longitude().get_units(), "degrees")
        self.assertEqual(result.get_latitude().get_value(), 36.51947777043374)
        self.assertEqual(result.get_latitude().get_units(), "degrees")

if __name__ == '__main__':
    unittest.main()



class TestResponseReader(unittest.TestCase):
    def setUp(self):
        self.completed_count = 0
        self.reader = joshpy.parse.ResponseReader(self.callback)

    def callback(self, count):
        self.completed_count = count

    def test_single_replicate(self):
        # Test data from a single replicate
        self.reader.process_response("[0] tree:height=10\tage=5\n")
        self.reader.process_response("[0] tree:height=12\tage=6\n")
        self.reader.process_response("[end 0]\n")

        results = self.reader.get_complete_replicates()
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 2)
        self.assertEqual(self.completed_count, 1)

        # Verify data structure
        first_point = results[0][0]
        self.assertEqual(first_point["target"], "tree")
        self.assertEqual(first_point["attributes"]["height"], 10)
        self.assertEqual(first_point["attributes"]["age"], 5)

    def test_multiple_replicates(self):
        # Test interleaved data from multiple replicates
        data = """[0] tree:height=10\tage=5
[1] tree:height=11\tage=5
[0] tree:height=12\tage=6
[1] tree:height=13\tage=6
[end 0]
[end 1]
"""
        self.reader.process_response(data)
        
        results = self.reader.get_complete_replicates()
        self.assertEqual(len(results), 2)
        self.assertEqual(len(results[0]), 2)
        self.assertEqual(len(results[1]), 2)
        self.assertEqual(self.completed_count, 2)

    def test_chunked_response(self):
        # Test handling of partial/chunked responses
        self.reader.process_response("[0] tree:height=10")
        self.reader.process_response("\tage=5\n[0] ")
        self.reader.process_response("tree:height=12\tage=6\n")
        self.reader.process_response("[end 0]\n")

        results = self.reader.get_complete_replicates()
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 2)
        self.assertEqual(self.completed_count, 1)

    def test_malformed_response(self):
        # Test handling of malformed responses
        with self.assertRaises(ValueError):
            self.reader.process_response("invalid\n")

    def test_empty_attributes(self):
        # Test handling of responses without attributes
        self.reader.process_response("[0] tree:\n")
        self.reader.process_response("[end 0]\n")

        results = self.reader.get_complete_replicates()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0]["target"], "tree")
        self.assertEqual(results[0][0]["attributes"], {})
