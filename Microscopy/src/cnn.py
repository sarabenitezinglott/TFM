import cv2
import sys
import numpy as np
import czifile
import skimage
import tifffile
from tqdm import tqdm
from scipy import ndimage as ndi
from skimage import morphology, filters
import matplotlib.pyplot as plt
import concurrent.futures
import keras
import tensorflow 
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, Cropping2D
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
                        cv2.imwrite(f'c:/Users/saraa/TFM/mask/nuclei/nuclei_mask_{i}.png', erode_uint8)
                    
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
                        cv2.imwrite(f'c:/Users/saraa/TFM/mask/nuclei/nuclei_mask_{i}.png', erode_uint8)

                    # elif j == 1: # Green channel --> citoplasm mask
                    #     cv2.imwrite(f'c:/Users/saraa/TFM/mask/citoplasm/cito_mask_{i}.png', erode_uint8)

                    # ### Not sure about this mask, shows everything, not just mitosis
                    # else:  # Blue channel --> mitosis mask
                    #     cv2.imwrite(f'c:/Users/saraa/TFM/mask/meiosis/meiosis_mask_{i}.png', erode_uint8)

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


# #### Model hyperparameters 
# # Learning rate
# LR = 0.0001
# # Custom loss function
# def dice_coef(y_true, y_pred):
#     smooth = 1.
#     y_true_f = keras.flatten(y_true)
#     y_pred_f = keras.flatten(y_pred)
#     intersection = keras.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)

# def bce_dice_loss(y_true, y_pred):
#     return 0.5 * tensorflow.keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

# epochs = 25
# batch = 8

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
    
def unet_model(input_layer, start_neurons):
    ''' Encoder layers or contraction patch'''
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1) 
    pool1 = Dropout(0.25)(pool1)
    print(conv1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)
    print(conv2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)
    print(conv3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    print(conv4)

    ''' Bridge '''
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
    
    ''' Decoder layer or expansion patch'''
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    print(deconv4)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    cropped_conv3 = Cropping2D(((0, 1), (0, 1)))(conv3) 
    print(cropped_conv3)
    uconv3 = concatenate([deconv3, cropped_conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    cropped_conv2 = Cropping2D(((0, 1), (0, 1)))(conv2)
    print(cropped_conv2)
    uconv2 = concatenate([deconv2, cropped_conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    cropped_conv1 = Cropping2D(((0, 1), (0, 1)))(conv1) 
    print(cropped_conv1)
    uconv1 = concatenate([deconv1, cropped_conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    return output_layer



## Before using it we need to use a compiler, to optimize it, for example Adam optimizer
# model = build_unet(input_shape)
# model.compile(optimizer=Adam(lr = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()