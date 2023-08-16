"""
copyright notice
This module is for testing code for neural network model.
"""
import os
from pathlib import Path


#############################################
# function to get full path string 
#############################################
def get_fullpath(path, filename):
    """
    To check path is file or directory  

    Args:
        path : file or directory path (string)
    Returns: string
        "directory" : if it is a directory
        "file" : if it is a file 
    """
    if path == "":
        return ""
    basefilename = os.path.basename(filename)
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)
    full_path = save_dir / basefilename
    return str(full_path)


#############################################
# function to check file is file or directory
#############################################
def check_file_type(filename):
    """
    To check path is file or directory  

    Args:
        path : file or directory path (string)
    Returns: string
        "directory" : if it is a directory
        "file" : if it is a file 
    """
    if filename in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        return "camera"

    if "://" in filename:
        return "url"

    if os.path.isdir(filename):
        return "directory"

    (fn, ext) = os.path.splitext(filename)
    if ext in [".mp4", ".avi", ".m4v", ".mpeg", ".flv", ".3gp",
            ".webm", ".mkv", ".mov"]:
        return "video"

    if ext in [".bmp", ".jpg", ".png", ".gif", ".jpeg", ".png",
            ".tiff", ".psd", ".ai"]:
        return "image"
    return "unknown"
