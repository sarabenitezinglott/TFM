import cv2
import czifile
import tifffile
import numpy as np
from tqdm import tqdm
import concurrent.futures
from scipy import ndimage as ndi
from skimage import morphology, filters
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, Activation, BatchNormalization
from keras.models import Model

#############################################
### Mask generation for each channel 


def mask(path, filetype):
    # Opening image with .czi file extension 
    if path.endswith(".czi"):
        data = czifile.imread(path)
        print(data.shape) # returns tuple of dimension sizes in STCZYX_ order (1, 201, 3, 1, 940, 940, 1)
        # Scene, Time, Channel, Slice, Height, Width
        time_points = data.shape[1]
        channel_count = data.shape[2]

    # Opening image with .tif file extension
    elif path.endswith(".tif"):
        data = tifffile.imread(path) 
        print(data.shape) # returns a tuple of dimensions TCHW  (201, 3, 940, 940)
        time_points = data.shape[0]
        channel_count = data.shape[1]

    # 0. Create a progress bar
    progress_bar = tqdm(total = time_points * channel_count)
    # 1. Loop for each image plane (adjust the indices as dependinf on czi and tif file extensions)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(time_points):
            for j in range(channel_count):
                # 1.1. Image plane if .czi file
                if filetype == "czi":
                    image_plane = data[0][i][j][0]
                    # 2. Convert the image to grayscale 
                    if data.shape[-1] == 3:
                        image_plane = cv2.cvtColor(image_plane,cv2.COLOR_BGR2GRAY)

                    # 3. Gaussian filter  
                    smooth = filters.gaussian(image_plane, sigma=4)

                    # 4. Otsu threshold application
                    threshold_value = filters.threshold_otsu(smooth)
                    thresh = smooth > threshold_value

                    # 5. Removing small and filling holes
                    fill = morphology.remove_small_holes(thresh, area_threshold= 70)

                    # 6. Perform dilation and erosion of the nucleus
                    dil = morphology.binary_dilation(fill)
                    erode = morphology.binary_erosion(dil)  # Boolean masks ('True' and 'False')
                    
                    # 7. Masks should be an 8-bit image: 'True' values become 0 and 'False' values 255
                    # 0 values correspond to the background, while 255 values correspond to the nuclei
                    erode_uint8 = np.uint8(erode) * 255

                    # 8. Save the mask for each channel in different folders
                    if j == 0: # Red channel --> nuclei mask 
                        cv2.imwrite(f'c:/Users/saraa/TFM/mask/nuclei/{i}.png', erode_uint8)
                    
                    progress_bar.update(1)

                # 1.2. Image plane if .tif file
                elif filetype == "tif":
                    image_plane = data[0][i][j]
                    # 2. Convert the image to grayscale 
                    if data.shape[-1] == 3:
                        image_plane = cv2.cvtColor(image_plane,cv2.COLOR_BGR2GRAY)

                    # 3. Gaussian filter  
                    smooth = filters.gaussian(image_plane, sigma=1.5)

                    # 4. Otsu threshold application
                    threshold_value = filters.threshold_otsu(smooth)
                    thresh = smooth < threshold_value

                    # 5. Fill holes
                    fill = ndi.binary_fill_holes(thresh)

                    # 6. Perform dilation and erosion of the nucleus
                    dil = morphology.binary_dilation(fill)
                    erode = morphology.binary_erosion(dil) 
                    
                    # 7. Masks should be an 8-bit image: 'True' values become 0 and 'False' values 255
                    erode_uint8 = np.uint8(erode) * 255

                    # 8. Save the mask for each channel in different folders
                    if j == 0: # Red channel --> nuclei mask 
                        cv2.imwrite(f'c:/Users/saraa/TFM/mask/nuclei/{i}.png', erode_uint8)

                    # elif j == 1: # Green channel --> citoplasm mask
                    #     cv2.imwrite(f'c:/Users/saraa/TFM/mask/citoplasm/{i}.png', erode_uint8)

                    # ### Not sure about this mask, shows everything, not just mitosis
                    # else:  # Blue channel --> mitosis mask
                    #     cv2.imwrite(f'c:/Users/saraa/TFM/mask/meiosis/{i}.png', erode_uint8)

                    progress_bar.update(1)

    progress_bar.close()
        

# def mean_iou(y_true, y_pred):
#     prec = []
#     for t in np.arange(0.5, 1.0, 0.05):
#         y_pred_ = tensorflow.to_int32(y_pred > t)
#         score, up_opt = tensorflow.metrics.mean_iou(y_true, y_pred_, 2)
#         keras.get_session().run(tensorflow.local_variables_initializer())
#         with tensorflow.control_dependencies([up_opt]):
#             score = tensorflow.identity(score)
#         prec.append(score)
#     return keras.mean(keras.stack(prec), axis=0)

# IMG_WIDTH       = 256
# IMG_HEIGHT      = 256
# IMG_CHANNELS    = 3

# print('Python       :', sys.version.split('\n')[0])
# print('Numpy        :', np.__version__)
# print('Skimage      :', skimage.__version__)
# print('Tensorflow   :', tensorflow.__version__)

##### U-NET ARCHITECTURE
# Maxpooling: It reduces half the image size
# Concatenation is important in U-Net, if not the segmentation would be imposible,
    # also, it would be a normal convolutional neural network. The concatenation means to add 
    # the filters obtained in the previous steps. So features are doubled (the previus one
    # and the new one). 
# To concatenate features, it must be the same lenght the convolutional layer to 
    # the decoder layer.   
# The pooling would be the output for the next layer. So: pool1, pool2, pool3 and pool 4
    # would be the outputs, but the inputs for the next layer.  
# The encoder blocks consists of a convolutional 2D layer (without batch normalization) 
    # with a maxpooling layer

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p   

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    ''' Input layer'''
    inputs = Input(input_shape)

    ''' Encoder layers or contraction patch'''
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    ''' Bridge '''
    b1 = conv_block(p4, 1024) 

    ''' Decoder layer or expansion patch'''
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  #Binary (can be multiclass)

    model = Model(inputs, outputs, name="U-Net")

    return model


