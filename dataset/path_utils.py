__all__ = [
    "get_paths_from_dirs",
]

import os 
from pathlib import Path
from typing import List, Dict
import glob

def get_paths_from_dirs(dirs: List[str], formats:List[str]) -> List[str]:
    if isinstance(dirs, str):
        dirs = [dirs]
    assert isinstance(formats, List)
    paths = []
    for dir in dirs:
        for formt in formats:
            root = "{}/**/*.{}".format(dir, formt)
            sub_paths = glob.glob(root,recursive=True)
            for sp in sub_paths:
                paths.append(Path(sp))
    return paths

def get_files_from_dirs(dirs:List[str]) -> List[str]:
    files = []
    for d in dirs:
        for f in os.listdir(d):
            files.append(Path(d,f))
    return files


