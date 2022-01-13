import os 
from pathlib import Path
from typing import List, Dict


def get_files(dirs:List[str]):
    files = []
    for d in dirs:
        for f in os.listdir(d):
            files.append(Path(d,f))
    return files
