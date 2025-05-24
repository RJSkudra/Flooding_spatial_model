import os

def get_dir(dir_name):
    """
    Returns the absolute path to a directory relative to the project root.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', dir_name))