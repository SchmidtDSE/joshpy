"""Structures to support virutal file systems for the Josh sandbox.

License: BSD-3-Clause
"""

import typing


class VirtualFile:
  """Definition of a file occupying a virutal file system.
  
  Definition of a file occupying a virutal file system which can be used in the sandbox Josh
  environment where access to the underlying file system is restricted.
  """

  def __init__(self, name: str, content: str, is_binary: bool):
    """Create a new record of a virtual file.
    
    Args:
      name: The name or path of the file which is in the virutal file system.
      content: The content of the file which is base64 encoded if the file is binary or plain text
        otherwise.
      is_binary: Flag indicating if the file is binary or plain text. If true, then the file is
        binary and its content is a string holding the base64 encoded version of its contents. If
        false, the file is plain text.
    """
    self._name = name
    self._content = content
    self._is_binary = is_binary
  
  def get_name(self) -> str:
    """Get the name or location of this file in the virtual file system.
    
    Returns:
      str: The name or path of the file which is in the virutal file system.
    """
    return self._name

  def get_content(self) -> str:
    """Get the contents of this file as a string.
    
    Returns:
      str: The content of the file which is base64 encoded if the file is binary or plain text
        otherwise.
    """
    return self._content

  def is_binary(self) -> bool:
    """Determine if this is a binary file.
    
    Returns:
      bool: Flag indicating if the file is binary or plain text. If true, then the file is binary
        and its content is a string holding the base64 encoded version of its contents. If false,
        the file is plain text.
    """
    return self._is_binary


def serialize_files(files: typing.List[VirtualFile]) -> str:
  """Serialize a virutal file system to a string representation.
  
  Args:
    files: The list of files in the virtual file system to be serialized.

  Returns:
    str: The string serialization of the given virtual file system in the format expected by the
      Josh server. Each file is represented as a string with tab-separated values in the format:
      filename\t(1 if binary, 0 if text)\tcontent\t. The content has tabs replaced with spaces
      and multiple files are concatenated without newlines.
  """
  serialized_files = []
  for file in files:
    safe_content = file.get_content().replace('\t', '    ')
    serialized_file = f"{file.get_name()}\t{1 if file.is_binary() else 0}\t{safe_content}\t"
    serialized_files.append(serialized_file)
  
  return ''.join(serialized_files)