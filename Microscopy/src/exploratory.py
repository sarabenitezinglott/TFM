import numpy as np
from aicspylibczi import CziFile
from pylibCZIrw import czi as pyczi
from pathlib import Path
import matplotlib.pyplot as plt

def openczi(path):
    czi = CziFile(path)
    return czi

def dimensions(czi):
    dims = czi.get_dims_shape() 
    return dims

def image_info(czi):
    img, shp = czi.read_image()
    return img, shp

