"""Utils for path manipulation.

This module contains utility functions for manipulationg
both directory and file paths.
"""
import os


def make_directory(directory_path: str) -> None:
    """Makes an empty directory if possible.
    
    Makes an empty directory if it is not presented.
    Returns if the directory exists.
    """
    try:
        os.mkdir(directory_path)
    except:
        return

def reset_file(file_path: str) -> None:
    """Resets the file at the given path.

    Tries to delete the file at the given path if it can be
    found there. Returns if the file can not be found at
    the given file path.

    Args:
        file_path:
            A str that specifies the path of the file.
    """
    try:
        os.remove(file_path)
    except:
        return