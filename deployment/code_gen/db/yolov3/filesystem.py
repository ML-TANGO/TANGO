import os
import shutil
from pathlib import Path


def make_dir(dir_path, use_base_path=True):
    """
    Creates given directory if not exists
    By default, it preappends use_base_path to the given dir_path
    """
    dir_path = (
        Path(f'{os.getenv("BASE_DIR")}{dir_path}') if use_base_path else Path(dir_path)
    )
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def file_from_bytes(file_bytes, dir_path, file_name):
    """
    Creates file from bytes, and saves it to dir_path
    """
    file_path = dir_path / str(file_name)
    shutil.copyfileobj(file_bytes, open(file_path, "wb"))
    return file_path
