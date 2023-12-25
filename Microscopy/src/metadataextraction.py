import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from czifile import CziFile
import os
import pydash
import ipywidgets as widgets
from matplotlib import pyplot as plt, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xmltodict
import numpy as np
from collections import Counter
from lxml import etree as ET
from aicsimageio import AICSImage
import dask.array as da
import napari
import pandas as pd

# Exploration 
def openczi(path):
    czi = CziFile(path)
    return czi

def dimensions(czi):
    dims = czi.get_dims_shape() 
    return dims

def image_info(czi):
    img, shp = czi.read_image()
    return img, shp


# Returns the type of the image based on the file extension - no magic

def get_imgtype(imagefile):
    imgtype = None

    if imagefile.lower().endswith('.ome.tiff') or imagefile.lower().endswith('.ome.tif'):
        # it is on OME-TIFF based on the file extension ... :-)
        imgtype = 'ometiff'
        print("This code does not support this image extension")

    elif imagefile.lower().endswith('.tiff') or imagefile.lower().endswith('.tif'):
        # it is on OME-TIFF based on the file extension ... :-)
        imgtype = 'tiff'
        print("This code does not support this image extension")

    elif imagefile.lower().endswith('.czi'):
        # it is on CZI based on the file extension ... :-)
        imgtype = 'czi'
        print("This file extension is code-supported")

    elif imagefile.lower().endswith('.png'):
        # it is on CZI based on the file extension ... :-)
        imgtype = 'png'
        print("This code does not support this image extension")

    elif imagefile.lower().endswith('.jpg') or imagefile.lower().endswith('.jpeg'):
        # it is on OME-TIFF based on the file extension ... :-)
        imgtype = 'jpg'
        print("This code does not support this image extension")

    return imgtype

# A dictionary will be created to hold the relevant metadata.

def create_metadata_dict():   
    metadata = {'Directory': None,
                'Filename': None,
                'Extension': None,
                'ImageType': None,
                'Name': None,
                'AcqDate': None,
                'TotalSeries': None,
                'SizeX': None,
                'SizeY': None,
                'SizeZ': None,
                'SizeC': None,
                'SizeT': None,
                'Sizes BF': None,
                'Axes': None,
                'Shape': None,
                'isRGB': None,
                'ObjNA': None,
                'ObjMag': None,
                'ObjID': None,
                'ObjName': None,
                'ObjImmersion': None,
                'XScale': None,
                'YScale': None,
                'ZScale': None,
                'XScaleUnit': None,
                'YScaleUnit': None,
                'ZScaleUnit': None,
                'DetectorModel': [],
                'DetectorName': [],
                'DetectorType': [],
                'DetectorID': [],
                'InstrumentID': None,
                'Channels': [],
                'ImageIDs': [],
                'NumPy.dtype': None
                }

    return metadata

# Returns a dictionary with metadata depending on the image type.
# Only CZI and OME-TIFF are currently supported.
# md = metadata ; additional_mdczi = additional metadata for czi

def get_metadata(imagefile, series=0):
    # get the image type
    imgtype = get_imgtype(imagefile)
    print('Image Type: ', imgtype)

    md = None
    additional_mdczi = None

    if imgtype == 'czi':

        # parse the CZI metadata return the metadata dictionary and additional information
        md = get_metadata_czi(imagefile, dim2none=False)
        additional_mdczi = get_additional_metadata_czi(imagefile)

    return md, additional_mdczi


def get_metadata_czi(filename, dim2none=False):
    # get CZI object and read array
    czi = CziFile(filename)

    # parse the XML into a dictionary
    metadatadict_czi = czi.metadata(raw=False)

    metadata = create_metadata_dict()
    print(type(metadata))

    # get directory and filename etc.
    metadata['Directory'] = os.path.dirname(filename)
    metadata['Filename'] = os.path.basename(filename)
    metadata['Extension'] = 'czi'
    metadata['ImageType'] = 'czi'

    # add axes and shape information using czifile package
    metadata['Axes'] = czi.axes
    metadata['Shape'] = czi.shape

    # add axes and shape information using aicsimageio package
    czi_aics = AICSImage(filename)
    metadata['Axes_aics'] = czi_aics.dims
    metadata['Shape_aics'] = czi_aics.shape  # returns tuple of dimension sizes in TCZYX order

    # determine pixel type for CZI array
    metadata['NumPy.dtype'] = czi.dtype

    # check if the CZI image is an RGB image depending on the last dimension entry of axes
    if czi.axes[-1] == 3:
        metadata['isRGB'] = True

    try:
        metadata['PixelType'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['PixelType']
    except KeyError as e:
        print('Key not found:', e)
        metadata['PixelType'] = None

    metadata['SizeX'] = int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeX'])
    metadata['SizeY'] = int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeY'])
    print(type(metadata))

    try:
        metadata['SizeZ'] = int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeZ'])
    except Exception as e:
        #print('Exception:', e)
        if dim2none:
            metadata['SizeZ'] = None
        if not dim2none:
            metadata['SizeZ'] = 1

    try:
        metadata['SizeC'] = int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeC'])
    except Exception as e:
        #print('Exception:', e)
        if dim2none:
            metadata['SizeC'] = None
        if not dim2none:
            metadata['SizeC'] = 1

    channels = []
    if metadata['SizeC'] == 1:
        try:
            channels.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                            ['Channels']['Channel']['ShortName'])
        except Exception as e:
            channels.append(None)

    if metadata['SizeC'] > 1:
        for ch in range(metadata['SizeC']):
            try:
                channels.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                                ['Channels']['Channel'][ch]['ShortName'])
            except Exception as e:
                print('Exception:', e)
                try:
                    channels.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                                    ['Channels']['Channel']['ShortName'])
                except Exception as e:
                    print('Exception:', e)
                    channels.append(str(ch))

    metadata['Channels'] = channels
    print(type(metadata))

    try:
        metadata['SizeT'] = int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeT'])
    except Exception as e:
        #print('Exception:', e)
        if dim2none:
            metadata['SizeT'] = None
        if not dim2none:
            metadata['SizeT'] = 1

    try:
        metadata['SizeM'] = int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeM'])
    except Exception as e:
        #print('Exception:', e)
        if dim2none:
            metadata['SizeM'] = None
        if not dim2none:
            metadata['SizeM'] = 1

    try:
        metadata['SizeB'] = int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeB'])
    except Exception as e:
        #print('Exception:', e)
        if dim2none:
            metadata['SizeB'] = None
        if not dim2none:
            metadata['SizeB'] = 1

    try:
        metadata['SizeS'] = int(metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['SizeS'])
    except Exception as e:
        print('Exception:', e)
        if dim2none:
            metadata['SizeS'] = None
        if not dim2none:
            metadata['SizeS'] = 1

    try:
        metadata['XScale'] = float(metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['Value']) * 1000000
        metadata['XScale'] = np.round(metadata['XScale'], 3)
        metadata['YScale'] = float(metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][1]['Value']) * 1000000
        metadata['YScale'] = np.round(metadata['YScale'], 3)
        try:
            metadata['XScaleUnit'] = metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['DefaultUnitFormat']
            metadata['YScaleUnit'] = metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][1]['DefaultUnitFormat']
        except KeyError as e:
            print('Key not found:', e)
            metadata['XScaleUnit'] = None
            metadata['YScaleUnit'] = None
        try:
            metadata['ZScale'] = float(metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][2]['Value']) * 1000000
            metadata['ZScale'] = np.round(metadata['ZScale'], 3)
            try:
                metadata['ZScaleUnit'] = metadatadict_czi['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][2]['DefaultUnitFormat']
            except KeyError as e:
                print('Key not found:', e)
                metadata['ZScaleUnit'] = metadata['XScaleUnit']
        except Exception as e:
            #print('Exception:', e)
            if dim2none:
                metadata['ZScale'] = None
                metadata['ZScaleUnit'] = None
            if not dim2none:
                # set to isotropic scaling if it was single plane only
                metadata['ZScale'] = metadata['XScale']
                metadata['ZScaleUnit'] = metadata['XScaleUnit']
    except Exception as e:
        print('Exception:', e)
        print('Scaling Data could not be found.')

    # try to get software version
    try:
        metadata['SW-Name'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Application']['Name']
        metadata['SW-Version'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Application']['Version']
    except KeyError as e:
        print('Key not found:', e)
        metadata['SW-Name'] = None
        metadata['SW-Version'] = None

    try:
        metadata['AcqDate'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['AcquisitionDateAndTime']
    except KeyError as e:
        print('Key not found:', e)
        metadata['AcqDate'] = None

    # get objective data
    try:
        metadata['ObjName'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective']['Name']
    except KeyError as e:
        print('Key not found:', e)
        metadata['ObjName'] = None

    try:
        metadata['ObjImmersion'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective']['Immersion']
    except KeyError as e:
        print('Key not found:', e)
        metadata['ObjImmersion'] = None

    try:
        metadata['ObjNA'] = float(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                     ['Instrument']['Objectives']['Objective']['LensNA'])
    except KeyError as e:
        print('Key not found:', e)
        metadata['ObjNA'] = None

    try:
        metadata['ObjID'] = metadatadict_czi['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective']['Id']
    except KeyError as e:
        print('Key not found:', e)
        metadata['ObjID'] = None

    try:
        metadata['TubelensMag'] = float(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                           ['Instrument']['TubeLenses']['TubeLens']['Magnification'])
    except KeyError as e:
        print('Key not found:', e)
        metadata['TubelensMag'] = None

    try:
        metadata['ObjNominalMag'] = float(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                             ['Instrument']['Objectives']['Objective']['NominalMagnification'])
    except KeyError as e:
        metadata['ObjNominalMag'] = None

    # get detector information
    # check if there are any detector entries inside the dictionary
    if pydash.objects.has(metadatadict_czi, ['ImageDocument', 'Metadata', 'Information', 'Instrument', 'Detectors']):

        if isinstance(metadatadict_czi['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'], list):
            num_detectors = len(metadatadict_czi['ImageDocument']['Metadata']['Information']['Instrument']['Detectors']['Detector'])
        else:
            num_detectors = 1

        # if there is only one detector found
        if num_detectors == 1:

            # check for detector ID
            try:
                metadata['DetectorID'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                              ['Instrument']['Detectors']['Detector']['Id'])
            except KeyError as e:
                metadata['DetectorID'].append(None)

            # check for detector Name
            try:
                metadata['DetectorName'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                ['Instrument']['Detectors']['Detector']['Name'])
            except KeyError as e:
                metadata['DetectorName'].append(None)

            # check for detector model
            try:
                metadata['DetectorModel'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                 ['Instrument']['Detectors']['Detector']['Manufacturer']['Model'])
            except KeyError as e:
                metadata['DetectorModel'].append(None)

            # check for detector type
            try:
                metadata['DetectorType'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                ['Instrument']['Detectors']['Detector']['Type'])
            except KeyError as e:
                metadata['DetectorType'].append(None)

        if num_detectors > 1:
            for d in range(num_detectors):
                # check for detector ID
                try:
                    metadata['DetectorID'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                  ['Instrument']['Detectors']['Detector'][d]['Id'])
                except KeyError as e:
                    metadata['DetectorID'].append(None)
                    
                # check for detector Name
                try:
                    metadata['DetectorName'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                    ['Instrument']['Detectors']['Detector'][d]['Name'])
                except KeyError as e:
                    metadata['DetectorName'].append(None)

                # check for detector model
                try:
                    metadata['DetectorModel'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                     ['Instrument']['Detectors']['Detector'][d]['Manufacturer']['Model'])
                except KeyError as e:
                    metadata['DetectorModel'].append(None)

                # check for detector type
                try:
                    metadata['DetectorType'].append(metadatadict_czi['ImageDocument']['Metadata']['Information']
                                                    ['Instrument']['Detectors']['Detector'][d]['Type'])
                except KeyError as e:
                    metadata['DetectorType'].append(None)


    # check for well information
    metadata['Well_ArrayNames'] = []
    metadata['Well_Indices'] = []
    metadata['Well_PositionNames'] = []
    metadata['Well_ColId'] = []
    metadata['Well_RowId'] = []
    metadata['WellCounter'] = []
    print(type(metadata))

    try:
        print('Trying to extract Scene and Well information if existing ...')
        # extract well information from the dictionary
        allscenes = metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['S']['Scenes']['Scene']

        # loop over all detected scenes
        for s in range(metadata['SizeS']):
            # more than one scene detected
            if metadata['SizeS'] > 1:
                # get the current well and add the array name to the metadata
                well = allscenes[s]
                metadata['Well_ArrayNames'].append(well['ArrayName'])

            # exactly one scene detected (e.g. after split scenes etc.)
            elif metadata['SizeS'] == 1:
                # only get the current well - no array names exist!
                well = allscenes  # fix here: use allscenes[0] instead of allscenes
                
            # get the well information
            for w in well:
                try:
                    metadata['Well_Indices'].append(w['Index'])
                except KeyError as e:
                    # print('Key not found in Metadata Dictionary:', e)
                    metadata['Well_Indices'].append(None)
                try:
                    metadata['Well_PositionNames'].append(w['Name'])
                except KeyError as e:
                    # print('Key not found in Metadata Dictionary:', e)
                    metadata['Well_PositionNames'].append(None)

        # more than one scene detected
        if metadata['SizeS'] > 1:
            # count the content of the list, e.g. how many times a certain well was detected
            metadata['WellCounter'] = Counter(metadata['Well_ArrayNames'])

        # exactly one scene detected (e.g. after split scenes etc.)
        elif metadata['SizeS'] == 1:
            # set ArrayNames equal to PositionNames for convenience
            metadata['Well_ArrayNames'] = metadata['Well_PositionNames']
            # count the content of the list, e.g. how many times a certain well was detected
            metadata['WellCounter'] = Counter(metadata['Well_PositionNames'])

        # count the number of different wells
        metadata['NumWells'] = len(metadata['WellCounter'].keys())

    except KeyError as e:
        print('No valid Scene or Well information found:', e)

    # close CZI file
    czi.close()
    print(type(metadata))

    return metadata


def get_additional_metadata_czi(filename):
    # get CZI object and read array
    czi = CziFile(filename)

    # parse the XML into a dictionary
    metadatadict_czi = xmltodict.parse(czi.metadata())
    additional_czimd = {}

    try:
        additional_czimd['Experiment'] = metadatadict_czi['ImageDocument']['Metadata']['Experiment']
    except:
        additional_czimd['Experiment'] = None

    try:
        additional_czimd['HardwareSetting'] = metadatadict_czi['ImageDocument']['Metadata']['HardwareSetting']
    except:
        additional_czimd['HardwareSetting'] = None

    try:
        additional_czimd['CustomAttributes'] = metadatadict_czi['ImageDocument']['Metadata']['CustomAttributes']
    except:
        additional_czimd['CustomAttributes'] = None

    try:
        additional_czimd['DisplaySetting'] = metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
    except KeyError as e:
        print('Key not found:', e)
        additional_czimd['DisplaySetting'] = None

    try:
        additional_czimd['Layers'] = metadatadict_czi['ImageDocument']['Metadata']['Layers']
    except KeyError as e:
        print('Key not found:', e)
        additional_czimd['Layers'] = None

    # close CZI file
    czi.close()

    return additional_czimd

# Convert the metadata dictionary to a Pandas DataFrame

def md2dataframe(metadata, paramcol='Parameter', keycol='Value'):
    mdframe = pd.DataFrame(columns=[paramcol, keycol])

    for k in metadata.keys():
        d = {'Parameter': k, 'Value': metadata[k]}
        df = pd.DataFrame([d], index=[0])
        mdframe = pd.concat([mdframe, df], ignore_index=True)

    return mdframe

def get_dimorder(dimstring):
    dimindex_list = []
    dims = ['R', 'I', 'M', 'H', 'V', 'B', 'S', 'T', 'C', 'Z', 'Y', 'X', '0']
    dims_dict = {}

    for d in dims:

        dims_dict[d] = dimstring.find(d)
        dimindex_list.append(dimstring.find(d))

    numvalid_dims = sum(i > 0 for i in dimindex_list)

    return dims_dict, dimindex_list, numvalid_dims

def get_array_czi(filename, metadata, 
                  replace_value=False,
                  remove_HDim=True,
                  return_addmd=False):

    # get CZI object and read array
    czi = CziFile(filename)
    cziarray = czi.asarray()

    # check for H dimension and remove
    if remove_HDim and metadata['Axes'][0] == 'H':
        metadata['Axes'] = metadata['Axes'][1:]
        cziarray = np.squeeze(cziarray, axis=0)

    # get additional information about dimension order etc.
    dim_dict, dim_list, numvalid_dims = get_dimorder(metadata['Axes'])
    metadata['DimOrder CZI'] = dim_dict

    if cziarray.shape[-1] == 3:
        pass
    else:
        cziarray = np.squeeze(cziarray, axis=len(metadata['Axes']) - 1)

    if replace_value:
        cziarray = replace_value(cziarray, value=0)

    # close czi file
    czi.close()

    return cziarray

# Displays the CZI or OME-TIFF image using a simple interactive viewer inside a Jupyter Notebook with dimension sliders

def display_image(array, metadata, sliders, b=0,
                  s=0, m=0, t=0, c=0, z=0,
                  vmin=0, vmax=1000):

    dim_dict = metadata[0]['DimOrder CZI']

    if metadata[0]['ImageType'] == 'ometiff':

        if sliders == 'TZCR':
            image = array[t - 1, z - 1, c - 1, :, :]

        if sliders == 'CTZR':
            image = array[c - 1, t - 1, z - 1, :, :]

        if sliders == 'TCZR':
            image = array[t - 1, c - 1, z - 1, :, :]

        if sliders == 'CZTR':
            image = array[c - 1, z - 1, t - 1, :, :]

        if sliders == 'ZTCR':
            image = array[z - 1, t - 1, c - 1, :, :]

        if sliders == 'ZCTR':
            image = array[z - 1, c - 1, z - 1, :, :]

    if metadata[0]['ImageType'] == 'czi':

        # add more dimension orders when needed
        if sliders == 'BTZCR':
            if metadata['isRGB']:
                image = array[b - 1, t - 1, z - 1, c - 1, :, :, :]
            else:
                image = array[b - 1, t - 1, z - 1, c - 1, :, :]

        if sliders == 'BTCZR':
            if metadata['isRGB']:
                image = array[b - 1, t - 1, c - 1, z - 1, :, :, :]
            else:
                image = array[b - 1, t - 1, c - 1, z - 1, :, :]

        if sliders == 'BSTZCR':
            if metadata['isRGB']:
                image = array[b - 1, s - 1, t - 1, z - 1, c - 1, :, :, :]
            else:
                image = array[b - 1, s - 1, t - 1, z - 1, c - 1, :, :]

        if sliders == 'BSTCZR':
            if metadata['isRGB']:
                image = array[b - 1, s - 1, t - 1, c - 1, z - 1, :, :, :]
            else:
                image = array[b - 1, s - 1, t - 1, c - 1, z - 1, :, :]

        if sliders == 'STZCR':
            if metadata['isRGB']:
                image = array[s - 1, t - 1, z - 1, c - 1, :, :, :]
            else:
                image = array[s - 1, t - 1, z - 1, c - 1, :, :]

        if sliders == 'STCZR':
            if metadata['isRGB']:
                image = array[s - 1, t - 1, c - 1, z - 1, :, :, :]
            else:
                image = array[s - 1, t - 1, c - 1, z - 1, :, :]

        if sliders == 'TZCR':
            if metadata['isRGB']:
                image = array[t - 1, z - 1, c - 1, :, :, :]
            else:
                image = array[t - 1, z - 1, c - 1, :, :]

        if sliders == 'TCZR':
            if metadata['isRGB']:
                image = array[t - 1, c - 1, z - 1, :, :, :]
            else:
                image = array[t - 1, c - 1, z - 1, :, :]

        if sliders == 'SCR':
            if metadata['isRGB']:
                image = array[s - 1, c - 1, :, :, :]
            else:
                image = array[s - 1, c - 1, :, :]

        if sliders == 'ZR':
            if metadata['isRGB']:
                image = array[z - 1, :, :, :]
            else:
                image = array[z - 1, :, :]

        if sliders == 'TR':
            if metadata['isRGB']:
                image = array[t - 1, :, :, :]
            else:
                image = array[t - 1, :, :]

        if sliders == 'CR':
            if metadata['isRGB']:
                image = array[c - 1, :, :, :]
            else:
                image = array[c - 1, :, :]

        if sliders == 'BSCR':
            if metadata['isRGB']:
                image = array[b - 1, s - 1, c - 1, :, :, :]
            else:
                image = array[b - 1, s - 1, c - 1, :, :]

        if sliders == 'BTCR':
            if metadata['isRGB']:
                image = array[b - 1, t - 1, c - 1, :, :, :]
            else:
                image = array[b - 1, t - 1, c - 1, :, :]

        ####### lightsheet Data #############
        if sliders == 'VIHRSCTZR':
            # reduce dimensions
            image = np.squeeze(array, axis=(0, 1, 2, 3, 4))
            image = image[c - 1, t - 1, z - 1, :, :]

    # display the labeled image
    fig, ax = plt.subplots(figsize=(8, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(image, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cm.gray)
    fig.colorbar(im, cax=cax, orientation='vertical')
    print('Min-Max (Current Plane):', image.min(), '-', image.max())

def create_ipyviewer_czi(cziarray, metadata):
    dim_dict = metadata['DimOrder CZI']

    useB = False
    useS = False

    if 'B' in dim_dict and dim_dict['B'] >= 0:
        useB = True
        b = widgets.IntSlider(description='Blocks:',
                              min=1,
                              max=metadata['SizeB'],
                              step=1,
                              value=1,
                              continuous_update=False)

    if 'S' in dim_dict and dim_dict['S'] >= 0:
        useS = True
        s = widgets.IntSlider(description='Scenes:',
                              min=1,
                              max=metadata['SizeS'],
                              step=1,
                              value=1,
                              continuous_update=False)

    t = widgets.IntSlider(description='Time:',
                          min=1,
                          max=metadata['SizeT'],
                          step=1,
                          value=1,
                          continuous_update=False)

    z = widgets.IntSlider(description='Z-Plane:',
                          min=1,
                          max=metadata['SizeZ'],
                          step=1,
                          value=1,
                          continuous_update=False)

    c = widgets.IntSlider(description='Channel:',
                          min=1,
                          max=metadata['SizeC'],
                          step=1,
                          value=1)

    print(cziarray.min(), cziarray.max())

    r = widgets.IntRangeSlider(description='Display Range:',
                               min=cziarray.min(),
                               max=cziarray.max(),
                               step=1,
                               value=[cziarray.min(), cziarray.max()],
                               continuous_update=False)

    # disable slider that are not needed
    if metadata['SizeB'] == 1 and useB:
        b.disabled = True
    if metadata['SizeS'] == 1 and useS:
        s.disabled = True
    if metadata['SizeT'] == 1:
        t.disabled = True
    if metadata['SizeZ'] == 1:
        z.disabled = True
    if metadata['SizeC'] == 1:
        c.disabled = True

    sliders = metadata['Axes'][:-3] + 'R'

    # TODO: this section is not complete, because it does not contain all possible cases
    # TODO: it is still under constrcution and can be done probably in a much smarter way

    if sliders == 'BTZCR':
        ui = widgets.VBox([b, t, z, c, r])

        def get_TZC_czi(b_ind, t_ind, z_ind, c_ind, r):
            display_image(cziarray, metadata, sliders, b=b_ind, t=t_ind, z=z_ind, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'b_ind': b, 't_ind': t, 'z_ind': z, 'c_ind': c, 'r': r})

    if sliders == 'BTCZR':
        ui = widgets.VBox([b, t, c, z, r])

        def get_TZC_czi(b_ind, t_ind, c_ind, z_ind, r):
            display_image(cziarray, metadata, sliders, b=b_ind, t=t_ind, c=c_ind, z=z_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'b_ind': b, 't_ind': t, 'c_ind': c, 'z_ind': z, 'r': r})

    if sliders == 'BSTZCR':
        ui = widgets.VBox([b, s, t, z, c, r])

        def get_TZC_czi(b_ind, s_ind, t_ind, z_ind, c_ind, r):
            display_image(cziarray, metadata, sliders, b=b_ind, s=s_ind, t=t_ind, z=z_ind, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'b_ind': b, 's_ind': s, 't_ind': t, 'z_ind': z, 'c_ind': c, 'r': r})

    if sliders == 'BSCR':
        ui = widgets.VBox([b, s, c, r])

        def get_TZC_czi(b_ind, s_ind, c_ind, r):
            display_image(cziarray, metadata, sliders, b=b_ind, s=s_ind, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'b_ind': b, 's_ind': s, 'c_ind': c, 'r': r})

    if sliders == 'BSTCZR':
        ui = widgets.VBox([b, s, t, c, z, r])

        def get_TZC_czi(b_ind, s_ind, t_ind, c_ind, z_ind, r):
            display_image(cziarray, metadata, sliders, b=b_ind, s=s_ind, t=t_ind, c=c_ind, z=z_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'b_ind': b, 's_ind': s, 't_ind': t, 'c_ind': c, 'z_ind': z, 'r': r})

    if sliders == 'STZCR':
        ui = widgets.VBox([s, t, z, c, r])

        def get_TZC_czi(s_ind, t_ind, z_ind, c_ind, r):
            display_image(cziarray, metadata, sliders, s=s_ind, t=t_ind, z=z_ind, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'s_ind': s, 't_ind': t, 'z_ind': z, 'c_ind': c, 'r': r})

    if sliders == 'STCZR':
        ui = widgets.VBox([s, t, c, z, r])

        def get_TZC_czi(s_ind, t_ind, c_ind, z_ind, r):
            display_image(cziarray, metadata, sliders, s=s_ind, t=t_ind, c=c_ind, z=z_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'s_ind': s, 't_ind': t, 'c_ind': c, 'z_ind': z, 'r': r})

    if sliders == 'TZCR':
        ui = widgets.VBox([t, z, c, r])

        def get_TZC_czi(t_ind, z_ind, c_ind, r):
            display_image(cziarray, metadata, sliders, t=t_ind, z=z_ind, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'t_ind': t, 'z_ind': z, 'c_ind': c, 'r': r})

    if sliders == 'TCZR':
        ui = widgets.VBox([t, c, z, r])

        def get_TZC_czi(t_ind, c_ind, z_ind, r):
            display_image(cziarray, metadata, sliders, t=t_ind, c=c_ind, z=z_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'t_ind': t, 'c_ind': c, 'z_ind': z, 'r': r})

    if sliders == 'SCR':
        ui = widgets.VBox([s, c, r])

        def get_TZC_czi(s_ind, c_ind, r):
            display_image(cziarray, metadata, sliders, s=s_ind, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'s_ind': s, 'c_ind': c, 'r': r})

    if sliders == 'ZR':
        ui = widgets.VBox([z, r])

        def get_TZC_czi(z_ind, r):
            display_image(cziarray, metadata, sliders, z=z_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'z_ind': z, 'r': r})

    if sliders == 'TR':
        ui = widgets.VBox([t, r])

        def get_TZC_czi(t_ind, r):
            display_image(cziarray, metadata, sliders, t=t_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'t_ind': t, 'r': r})

    if sliders == 'CR':
        ui = widgets.VBox([c, r])

        def get_TZC_czi(c_ind, r):
            display_image(cziarray, metadata, sliders, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'c_ind': c, 'r': r})

    if sliders == 'BTCR':
        ui = widgets.VBox([b, t, c, r])

        def get_TZC_czi(b_ind, t_ind, c_ind, r):
            display_image(cziarray, metadata, sliders, b=b_ind, t=t_ind, c=c_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'b_ind': b, 't_ind': t, 'c_ind': c, 'r': r})

    ############### Lightsheet data #################

    if sliders == 'VIHRSCTZR':
        ui = widgets.VBox([c, t, z, r])

        def get_TZC_czi(c_ind, t_ind, z_ind, r):
            display_image(cziarray, metadata, sliders, c=c_ind, t=t_ind, z=z_ind, vmin=r[0], vmax=r[1])

        out = widgets.interactive_output(get_TZC_czi, {'c_ind': c, 't_ind': t, 'z_ind': z, 'r': r})

    return out, ui


# Get the pixel data of the CZI file as multidimensional NumPy.Array

def get_array_pylibczi(filename, return_addmd=False, **kwargs):
    metadata = get_metadata_czi(filename)
    additional_metadata_czi = get_additional_metadata_czi(filename)


def get_scalefactor(metadata):
    """Add scaling factors to the metadata dictionary

    :param metadata: dictionary with CZI or OME-TIFF metadata
    :type metadata: dict
    :return: dictionary with additional keys for scling factors
    :rtype: dict
    """

    # set default scale factore to 1
    scalefactors = {'xy': 1.0,
                    'zx': 1.0
                    }

    try:
        # get the factor between XY scaling
        scalefactors['xy'] = np.round(metadata['XScale'] / metadata['YScale'], 3)
        # get the scalefactor between XZ scaling
        scalefactors['zx'] = np.round(metadata['ZScale'] / metadata['YScale'], 3)
    except KeyError as e:
        print('Key not found: ', e)

    return scalefactors


def show_napari(array, metadata,
                blending='additive',
                gamma=0.85,
                verbose=True,
                use_pylibczi=True):
    """Show the multidimensional array using the Napari viewer

    :param array: multidimensional NumPy.Array containing the pixeldata
    :type array: NumPy.Array
    :param metadata: dictionary with CZI or OME-TIFF metadata
    :type metadata: dict
    :param blending: NapariViewer option for blending, defaults to 'additive'
    :type blending: str, optional
    :param gamma: NapariViewer value for Gamma, defaults to 0.85
    :type gamma: float, optional
    :param verbose: show additional output, defaults to True
    :type verbose: bool, optional
    :param use_pylibczi: specify if pylibczi was used to read the CZI file, defaults to True
    :type use_pylibczi: bool, optional
    """

    def calc_scaling(data, corr_min=1.0,
                     offset_min=0,
                     corr_max=0.85,
                     offset_max=0):

        # get min-max values for initial scaling
        minvalue = np.round((data.min() + offset_min) * corr_min)
        maxvalue = np.round((data.max() + offset_max) * corr_max)
        print('Scaling: ', minvalue, maxvalue)

    with napari.gui_qt():

        # create scalefcator with all ones
        scalefactors = [1.0] * len(array.shape)

        # initialize the napari viewer
        print('Initializing Napari Viewer ...')
        viewer = napari.Viewer()

        if metadata['ImageType'] == 'ometiff':

            # find position of dimensions
            posZ = metadata['DimOrder BF Array'].find('Z')
            posC = metadata['DimOrder BF Array'].find('C')
            posT = metadata['DimOrder BF Array'].find('T')

            # get the scalefactors from the metadata
            scalef = get_scalefactor(metadata)
            # modify the tuple for the scales for napari
            scalefactors[posZ] = scalef['zx']

            if verbose:
                print('Dim PosT : ', posT)
                print('Dim PosC : ', posC)
                print('Dim PosZ : ', posZ)
                print('Scale Factors : ', scalefactors)

            # add all channels as layers
            for ch in range(metadata['SizeC']):

                try:
                    # get the channel name
                    chname = metadata['Channels'][ch]
                except:
                    # or use CH1 etc. as string for the name
                    chname = 'CH' + str(ch + 1)

                # cutout channel
                channel = array.take(ch, axis=posC)
                print('Shape Channel : ', ch, channel.shape)

                # actually show the image array
                print('Scaling Factors: ', scalefactors)

                # get min-max values for initial scaling
                clim = [channel.min(), np.round(channel.max() * 0.85)]
                if verbose:
                    print('Scaling: ', clim)
                viewer.add_image(channel,
                                 name=chname,
                                 scale=scalefactors,
                                 contrast_limits=clim,
                                 blending=blending,
                                 gamma=gamma)

        if metadata['ImageType'] == 'czi':

            if not use_pylibczi:
                # use find position of dimensions
                posZ = metadata['Axes'].find('Z')
                posC = metadata['Axes'].find('C')
                posT = metadata['Axes'].find('T')

            if use_pylibczi:
                posZ = metadata['Axes_aics'].find('Z')
                posC = metadata['Axes_aics'].find('C')
                posT = metadata['Axes_aics'].find('T')

            # get the scalefactors from the metadata
            scalef = get_scalefactor(metadata)
            # modify the tuple for the scales for napari
            scalefactors[posZ] = scalef['zx']

            if verbose:
                print('Dim PosT : ', posT)
                print('Dim PosZ : ', posZ)
                print('Dim PosC : ', posC)
                print('Scale Factors : ', scalefactors)

            if metadata['SizeC'] > 1:
                # add all channels as layers
                for ch in range(metadata['SizeC']):

                    try:
                        # get the channel name
                        chname = metadata['Channels'][ch]
                    except:
                        # or use CH1 etc. as string for the name
                        chname = 'CH' + str(ch + 1)

                    # cut out channel
                    # use dask if array is a dask.array
                    if isinstance(array, da.Array):
                        print('Extract Channel using Dask.Array')
                        channel = array.compute().take(ch, axis=posC)

                    else:
                        # use normal numpy if not
                        print('Extract Channel NumPy.Array')
                        channel = array.take(ch, axis=posC)

                    print('Shape Channel : ', ch, channel.shape)

                    # actually show the image array
                    print('Adding Channel: ', chname)
                    print('Scaling Factors: ', scalefactors)

                    # get min-max values for initial scaling
                    # clim = calc_scaling(channel)

                    viewer.add_image(channel,
                                     name=chname,
                                     scale=scalefactors,
                                     # contrast_limits=clim,
                                     blending=blending,
                                     gamma=gamma,
                                     is_pyramid=False)

            if metadata['SizeC'] == 1:

                # just add one channel as a layer
                try:
                    # get the channel name
                    chname = metadata['Channels'][0]
                except:
                    # or use CH1 etc. as string for the name
                    chname = 'CH' + str(ch + 1)

                # actually show the image array
                print('Adding Channel: ', chname)
                print('Scaling Factors: ', scalefactors)

                # get min-max values for initial scaling
                # clim = calc_scaling(array)

                viewer.add_image(array,
                                 name=chname,
                                 scale=scalefactors,
                                 # contrast_limits=clim,
                                 blending=blending,
                                 gamma=gamma,
                                 is_pyramid=False)


def check_for_previewimage(czi):
    """Check if the CZI contains an image from a prescan camera

    :param czi: CZI imagefile object
    :type metadata: CziFile object
    :return: has_attimage - Boolean if CZI image contains prescan image
    :rtype: bool
    """

    att = []

    for attachment in czi.attachments():
        entry = attachment.attachment_entry
        print(entry.name)
        att.append(entry.name)

    has_attimage = False

    if 'SlidePreview' in att:
        has_attimage = True

    return has_attimage


def writexml_czi(filename, xmlsuffix='_CZI_MetaData.xml'):
    # open czi file and get the metadata
    czi = CziFile(filename)
    mdczi = czi.metadata()
    czi.close()

    # change file name
    xmlfile = filename.replace('.czi', xmlsuffix)

    # get tree from string
    tree = ET.ElementTree(ET.fromstring(mdczi))

    # write XML file to same folder
    tree.write(xmlfile, encoding='utf-8', method='xml')

    return xmlfile
