from src import metadataextraction as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO

###### Extracting metadata from czi
# Using metadataextraction notebook's functions

'''
Understanding .czi extension 

For this aim, **metadata information** from each czi has to be extracted.
In Windows different softwares can be used for this apporach, such as: **Image J**, ZEN or NetScope.  

3288-1.czi file contains:
- 3 channels:
    {   c0
            NamemCher-T1/Name
            Index0/Index
            ChannelProcesssingModeProcessChannel/ChannelProcessingMode
            Represents: 
                Pcn1: nuclei -> DNA synthesis               
                RitC: membrane

        /c0
        c1
            NamemNeonGreen-T2/Name
            Index1/Index
            ChannelProcessingModeProcessChannel/ChannelProcessingMode
            Represents: 
                SynCut3: translocation to nucleus -> mitosis entry
        /c1
        c2
            NameEBFP2-T3/Name
            Index2/Index
            ChannelProcessingModeProcessChannel/ChannelProcessingMode
            Represents: 
                Sid2: spindle pole body and early separation
        /c2
    }

------------------------
Information CZI Dimension Characters:
    - '0': 'Sample',  # e.g. RGBA
    - 'X': 'Width',
    - 'Y': 'Height',
    - 'C': 'Channel',
    - 'Z': 'Slice',  # depth
    - 'T': 'Time',
    - 'R': 'Rotation',
    - 'S': 'Scene',  # contiguous regions of interest in a mosaic image
    - 'I': 'Illumination',  # direction
    - 'B': 'Block',  # acquisition
    - 'M': 'Mosaic',  # index of tile for compositing a scene
    - 'H': 'Phase',  # e.g. Airy detector fibers
    - 'V': 'View',  # e.g. for SPIM

'''

# 1. Exploring czi file
def czi(filepath):
    czi = md.openczi(filepath)
    # Image dimension
    dims = md.dimensions(czi)
    print(dims)
    # Channel information
    first_channel, second_channel, third_channel = md.image_info(czi)
    # Metadata function from czifile library
    xml_metadata = md.basic_metadata(filepath)
    # Image timeseries
    timeseries = md.timeseries(filepath)
    # Open image
    img = md.img(filepath)

filepath = "D:/TFM/Microscopy/video/3288-1-AP-OP.czi"
czi()


# 2. Extraction of metadata and dataframe creation
def czi3288():
    # Image type
    img_type = md.get_imgtype("3288-1-AP-OP.czi")
    print(f'Image type:', img_type)
    # Metadata
    metadata = md.get_metadata("D:/TFM/Microscopy/video/3288-1-AP-OP.czi")
    print(f'Object data type is:', type(metadata))
    print(metadata[0])
    print(metadata[1])
    # Additional information
    md1 = md.get_additional_metadata_czi("D:/TFM/Microscopy/video/3288-1-AP-OP.czi")
    print(f'Additional metadata:', md1)
    # Convert information into a dataframe 
    df= md.md2dataframe(metadata[0], paramcol='Parameter', keycol='Value')
    print(df)
    # More information 
    czi_array = md.get_array_czi("D:/TFM/Microscopy/video/3288-1-AP-OP.czi", metadata[0])
    print(f'Czi array:', czi_array)
    # XML format
    md.writexml_czi("D:/TFM/Microscopy/video/3288-1-AP-OP.czi", xmlsuffix='_CZI_MetaData.xml')

    return metadata

metadata = czi3288()
