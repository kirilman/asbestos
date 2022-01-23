import os 
from pathlib import Path
from typing import List, Dict


def get_files_from_dirs(dirs:List[str]) -> List[str]:
    files = []
    for d in dirs:
        for f in os.listdir(d):
            files.append(Path(d,f))
    return files
