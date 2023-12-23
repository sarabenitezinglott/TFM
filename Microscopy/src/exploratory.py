import numpy as np
from aicspylibczi import CziFile
from pathlib import Path
import matplotlib.pyplot as plt

def openczi(path):
    czi = CziFile(path)
    return czi

def czimetadata(czi):
    dimensions = czi.get_dims_shape() 
    print(czi.dims)
    print(czi.size)
    #img, shp = czi.read_image(S=7, Z=0)
    img, shp = czi.read_image()
    return dimensions, img, shp


def norm_by(x, min_, max_):
    norms = np.percentile(x, [min_, max_])
    i2 = np.clip((x - norms[0]) / (norms[1] - norms[0]), 0, 1)
    return i2

def recolor(im):  # transform from rgb to cyan-magenta-yellow
    im_shape = np.array(im.shape)
    color_transform = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]).T
    im_reshape = im.reshape([np.prod(im_shape[0:2]), im_shape[2]]).T
    im_recolored = np.matmul(color_transform.T, im_reshape).T
    im_shape[2] = 3
    im = im_recolored.reshape(im_shape)
    return im

def img_plot(cmy):
    plt.figure(figsize=(10, 10))
    plt.imshow(cmy)
    plt.axis('off')
