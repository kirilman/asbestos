from .path_utils import get_paths_from_dirs
from pathlib import Path
import os
from dataset import PathLike
from abc import abstractmethod, ABC
from pycocotools.coco import COCO
import pandas as pd
import json

class Extractor:
    def __init__(path_for_processing: PathLike):
        paths = get_paths_from_dirs()


class FileProcessing(ABC):
    def __init__(self, file):
        self.file = file
        pass

    @abstractmethod
    def process(self):
        pass

def read_pandas_segmentation(path_2_json):
    return pd.DataFrame({d['id']: d for d in json.load(open(path_2_json,'r'))['annotations']}).T
    

class JsonSegmentProcessing(FileProcessing):
    def __init__(self, file):
        super().__init__(file)

    def process(self):
        segments = self.file.readlines()
        
        return

    def 