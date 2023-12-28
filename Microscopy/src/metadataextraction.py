import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from aicsimageio.readers import CziReader
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
    czi = CziReader(path)
    return czi

def dimensions(czi):
    data = czi.data
    dims = czi.shape
    return dims

def image_info(czi):
    first_channel = czi.get_image_data("ZYX", C=0, S=0, T=0)
    second_channel = czi.get_image_data("ZYX", C=1, S=0, T=0)
    third_channel = czi.get_image_data("ZYX", C=2, S=0, T=0)
    return first_channel, second_channel, third_channel

def basic_metadata(filepath):
    with CziFile(filepath) as czi:
        xml_metadata = czi.metadata()
    return xml_metadata

def timeseries(filepath):
    timestamps = None 
    with CziFile(filepath) as czi:
        for attachment in czi.attachments():
            if attachment.attachment_entry.name == 'TimeStamps':
                timestamps = attachment.data()
                break
    return timestamps

def img(filepath):
    with CziFile(filepath) as czi:
        img = czi.asarray()
    return img

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
