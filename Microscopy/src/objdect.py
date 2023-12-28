import numpy as np
import napari
from aicsimageio import AICSImage

# Visualize the images: 
def visualize_images(filepath):
    aics_image = AICSImage(filepath)
    # Open Napari Viewer
    viewer = napari.Viewer()

    # Add your image to the viewer
    viewer.add_image(aics_image.data)

    # Run the Napari viewer
    napari.run()

def napari_view_channels():
    image_paths = ["images/red.png", "images/blue.png", "images/green.png"]
    for i in image_paths:
        print(i)