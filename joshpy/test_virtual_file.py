import typing
import unittest

import joshpy.virtual_file


class TestVirtualFile(unittest.TestCase):
    def setUp(self):
        self.text_file = joshpy.virtual_file.VirtualFile("hello.txt", "Hello World", False)
        self.binary_file = joshpy.virtual_file.VirtualFile("test.bin", "YmluYXJ5", True)
        
    def test_get_name(self):
        self.assertEqual(self.text_file.get_name(), "hello.txt")
        self.assertEqual(self.binary_file.get_name(), "test.bin")

    def test_get_content(self):
        self.assertEqual(self.text_file.get_content(), "Hello World")
        self.assertEqual(self.binary_file.get_content(), "YmluYXJ5")

    def test_is_binary(self):
        self.assertFalse(self.text_file.is_binary())
        self.assertTrue(self.binary_file.is_binary())

    def test_serialize_files_empty(self):
        files: typing.List[joshpy.virtual_file.VirtualFile] = []
        result = joshpy.virtual_file.serialize_files(files)
        self.assertEqual(result, "")

    def test_serialize_single_text_file(self):
        files = [self.text_file]
        result = joshpy.virtual_file.serialize_files(files)
        expected = "hello.txt\t0\tHello World\t"
        self.assertEqual(result, expected)

    def test_serialize_single_binary_file(self):
        files = [self.binary_file]
        result = joshpy.virtual_file.serialize_files(files)
        expected = "test.bin\t1\tYmluYXJ5\t"
        self.assertEqual(result, expected)

    def test_serialize_multiple_files(self):
        files = [self.text_file, self.binary_file]
        result = joshpy.virtual_file.serialize_files(files)
        expected = "hello.txt\t0\tHello World\ttest.bin\t1\tYmluYXJ5\t"
        self.assertEqual(result, expected)

    def test_serialize_with_tabs_in_content(self):
        file_with_tabs = joshpy.virtual_file.VirtualFile("tabs.txt", "Hello\tWorld", False)
        result = joshpy.virtual_file.serialize_files([file_with_tabs])
        expected = "tabs.txt\t0\tHello    World\t"
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
