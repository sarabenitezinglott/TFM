import cv2
import sys
import numpy as np
import czifile
import skimage
from tqdm import tqdm
from scipy import ndimage as ndi
from skimage import morphology, filters
import matplotlib.pyplot as plt
import concurrent.futures
import keras
import tensorflow 
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from keras.models import Model

#############################################
### Mask generation for each channel 


def mask(czi_path):
    # Load CZI file
    czi_data = czifile.imread(czi_path)

    print(czi_data.shape) # returns tuple of dimension sizes in STCZYX order
    # Scene, Time, Channel, Slice, Height, Width
    time_points = czi_data.shape[1]
    channel_count = czi_data.shape[2]
    # 0. Create a progress bar
    progress_bar = tqdm(total = time_points * channel_count)
    # Select the desired image plane (adjust the indices as needed)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(czi_data.shape[1]):
            for j in range(czi_data.shape[2]):
                image_plane = czi_data[0][i][j][0]  

                # 2. Convert the image to grayscale 
                if image_plane.shape[-1] == 3:
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

                # 7. Save the mask for each channel in different folders
                if j == 0: # Red channel --> nuclei mask 
                    plt.imshow(erode, cmap='binary')
                    plt.axis('off')
                    plt.savefig(f'c:/Users/saraa/TFM/mask/nuclei/nuclei_mask_{i}.png')

                elif j == 1: # Green channel --> citoplasm mask
                    fill = ndi.binary_fill_holes(erode)
                    plt.imshow(fill, cmap='binary')
                    plt.axis('off')
                    plt.savefig(f'c:/Users/saraa/TFM/mask/citoplasm/cito_mask_{i}.png')

                ### Not sure about this mask, shows everything, not just mitosis
                else:  # Blue channel --> mitosis mask
                    plt.imshow(erode, cmap='binary')
                    plt.axis('off')
                    plt.savefig(f'c:/Users/saraa/TFM/mask/meiosis/meiosis_mask_{i}.png')

                progress_bar.update(1)

    progress_bar.close()


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tensorflow.to_int32(y_pred > t)
        score, up_opt = tensorflow.metrics.mean_iou(y_true, y_pred_, 2)
        keras.get_session().run(tensorflow.local_variables_initializer())
        with tensorflow.control_dependencies([up_opt]):
            score = tensorflow.identity(score)
        prec.append(score)
    return keras.mean(keras.stack(prec), axis=0)

IMG_WIDTH       = 256
IMG_HEIGHT      = 256
IMG_CHANNELS    = 3

print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Skimage      :', skimage.__version__)
print('Tensorflow   :', tensorflow.__version__)


#### Model hyperparameters 
# Learning rate
LR = 0.0001
# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * tensorflow.keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

epochs = 25
batch = 8

##### U-NET ARCHITECTURE
    
def build_model(input_layer, start_neurons):
    ''' Encoder layer '''
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    ''' Bridge '''
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
    
    ''' Decoder layer'''
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    return output_layer

input_layer = Input((100, 100, 1))
output_layer = build_model(input_layer, 16)