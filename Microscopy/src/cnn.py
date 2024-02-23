import os
import cv2
import random
import czifile
import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from keras.models import Model
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology, filters
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, Activation, BatchNormalization

# 1. Mask generation for the U-net model 

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
    

# 2. U-NET model architecture
''' 
Maxpooling is needed to reduce half the image size.
Concatenation is important in U-Net, if not the segmentation would be impossible,
    also, it would be a normal convolutional neural network. The concatenation means to add 
    the filters obtained in the previous steps. So features are doubled (the previus one
    and the new one). 
To concatenate features, it must be the same lenght the convolutional layer to 
    the decoder layer.   
The pooling would be the output for the next layer. So: pool1, pool2, pool3 and pool 4
    would be the outputs, but the inputs for the next layer.  
The encoder blocks consists of a convolutional 2D layer (without batch normalization) 
    with a maxpooling layer 

''' 
# To optimice the model, is necessary to make a function foreach block in 
    # the model. 
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

# 3. Model fitting
    # EalryStopping will stop the training when there is no imporvement in the 
    # validation loss for three consecutives epochs
def fitting(model, train_generator, valid_generator, weightpath, tensorboard):
    # Improving model fitting 
    checkpoints = ModelCheckpoint(filepath = weightpath, save_weigths_only = True, 
                                  monitor = 'val_accuracy', mode = 'max', 
                                  save_best_only = True)

    callbacks = [EarlyStopping(monitor = 'val_loss', patience = 3),
                TensorBoard(log_dir = tensorboard), checkpoints]
    steps_per_epoch = 60
    history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, 
                    validation_data = valid_generator, validation_steps=6, epochs = 3,
                    callbacks= callbacks, verbose = 1)
    return history

# 4. Model predicting
def predict(model, Xtrain, Xtest, Ytrain):
    # idx = random.randint(0, len(Xtrain))
    pred_train = model.predict(Xtrain[:int(Xtrain.shape[0]*0.9)], verbose = 1)
    pred_train_t = (pred_train > 0.5).astype(np.uint8)
    pred_val = model.predict(Xtrain[int(Xtrain.shape[0]*0.9):], verbose = 1)
    pred_val_t = (pred_val > 0.5).astype(np.uint8)
    pred_test = model.predict(Xtest, verbose = 1)
    pred_test_t = (pred_test > 0.5).astype(np.uint8)

    # Plots on training sample
    ix = random.randint(0, len(pred_train_t))
    plt.imshow(Xtrain[ix]) 
    plt.show()
    plt.imshow(np.squeeze(Ytrain[ix]))
    plt.show()
    plt.imshow(np.squeeze(pred_train_t[ix]))
    plt.show()

    # Plots on validation sample
    ix = random.randint(0, len(pred_val_t))
    plt.imshow(Xtrain[int(Xtrain.shape[0]*0.9):][ix]) 
    plt.show()
    plt.imshow(np.squeeze(Ytrain[int(Ytrain.shape[0]*0.9):][ix]))
    plt.show()
    plt.imshow(np.squeeze(pred_val_t[ix]))
    plt.show()

    return pred_train_t, pred_val_t, pred_test_t

# 5. Count the number of nuclei
def nuclei_segmentation(folder):
    results = [] 

    # 1. Opening each mask file
    for img in os.scandir(folder):
        img_path = os.path.join(folder, img)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # 2. Verify if the image is loaded correctly
        if image is None:
            raise ValueError(f"Unable to load image: {img_path}")
        # 3. Apply again thresholding. so values are 0 or 255. 127 is the
            #  threshold. Lower than 127 is 0, higher is 255
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # 4. Connected components analysis
        num_labels, _ = cv2.connectedComponents(binary_image)
        num_blobs = num_labels - 1
        
        results.append({'Image_path': img_path, 'Number of Nuclei': num_blobs})
    
    # 5. Creating DataFrame from the results list. 
    df = pd.DataFrame(results)
    df["Timepoint"] = df["Image_path"].str.extract(r'(\d+)').astype(int)
    df.sort_values(by="Timepoint", inplace = True)
    df.set_index("Timepoint", inplace=True)
    df.drop(columns=["Image_path"], inplace=True)

    return df