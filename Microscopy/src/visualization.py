import numpy as np
import napari
from aicsimageio import AICSImage
from IPython.display import display, Image

# Visualize the images. Channels are mixed
def visualize_images(filepath):
    aics_image = AICSImage(filepath)
    # Open Viewer
    viewer = napari.Viewer()

    # Add image to the viewer
    viewer.add_image(aics_image.data, channel_axis=1, name=["Pcn1 & RitC", "SynCut3", "Sid2"],
                     colormap=["red","green","blue"],  contrast_limits=[[1000, 10308], [1000, 8773], [1000, 5990]], 
                     gamma = [0.82, 0.56, 0.40])

    # Napari viewer
    napari.run()

def napari_view_splited_channels():
    image_paths = ["C:/Users/saraa/TFM/images/red.png", "C:/Users/saraa/TFM/images/green.png", "C:/Users/saraa/TFM/images/blue.png"]
    for i in image_paths:
        display(Image(i, width = 500))


