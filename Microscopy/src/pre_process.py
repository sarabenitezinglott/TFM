import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import cv2
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import io
from sklearn.model_selection import train_test_split

# 1. Create dataframes 
# 1.1 Create dataframe containing the .tiff images of each folder. A dataframe for each folder, so split can be done. 
# dfs[0] is blue, dfs[1] is green, dfs[2] is red

def create_df(folder):
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    print(subfolders)
    dfs = []

    for sub in subfolders:
        list_tiff = []

        for file in os.listdir(sub):
            list_tiff.append(os.path.join(sub, file))
        
        df = pd.DataFrame(list_tiff, columns=["File_path"])
            # 1. File path
        df["File_path1"] = df["File_path"].str.extract(r'D:/TFM/Microscopy/video/copies/channels/(.*)')
        df["File_path1"] = df["File_path1"].str.replace("\\","/")
            # 2. Image ID
        df["Image_id"] = df["File_path1"].str.findall(r'(\d{4}-\d{1}[a-z]\d{4})').str[0]
            # 3. Channel and copie columns
        df[["Channel", "Copie"]] = df["File_path1"].str.split("/", expand=True)
            # 4. Extraction of copie information for three different columns
        df[["Sample", "Duplicates","Timepoints"]] = df["Copie"].str.extract(r'(\d{4})-(\d{1})[a-z](\d{4})')
        df["Timepoints"] = df["Timepoints"].astype(int)
            # 5. Cleaning
        df.drop(columns=["File_path1", "Copie"], inplace=True)
            # 6. Label 
        lista = ["red","green", "blue"]
        for color in lista:
            df.loc[df["Channel"] == color, "Label"] = lista.index(color)
        df["Label"] = df["Label"].astype(int)
        dfs.append(df)

    blue, green, red = dfs[0], dfs[1], dfs[2]
    
    # Concatenate df vertically. pd.merge() is not an option as it combines df horizontally.
    semi = pd.concat([blue, green], axis=0)
    all_df = pd.concat([red, semi], axis=0)
    all_df = all_df.reset_index(drop=True)

    return blue, green, red, all_df

# 1.2 Create dataframes for mask
def mask_df(folder):
    list_tiff = []
    for file in os.listdir(folder):
        list_tiff.append(os.path.join(folder, file))
    # Create df
        # 1. File path
    df1 = pd.DataFrame(list_tiff, columns=["File_path"])
        # 2. Type of image (nuclei, cytoplasm, meiosis)
    df1["Type"] = df1["File_path"].str.extract(r'(nuclei/)')
        # 3. Timepoints
    df1["Timepoint"] =  df1["File_path"].str.extract(r'(\d+)')
    df1["Image_id"] = df1["Type"] + df1["Timepoint"]
    df1["Image_id"] = df1["Image_id"].replace("/", "_")
        # 4. Channel 
    if df1["Image_id"].str.contains("nuclei").any():
        df1["Channel"] = "red"
    elif df1["Image_id"].str.contains("citoplasm").any():
        df1["Channel"] = "green"
    else:
        df1["Channel"] = "blue"
        # 5. Label
    label_dict = {'red': 0, 'green': 1, 'blue': 2}
    df1['Label'] = df1['Channel'].map(label_dict)
    df1.sort_values(by="Timepoint", inplace = True)
    df1 = df1.reset_index(drop = True)
    return df1

# 2. Split data into images and masks sets for U-Net
def train_valid_split(df_images, df_mask):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(df_images,df_mask, test_size=0.2, random_state=0)
    return Xtrain, Xtest, Ytrain, Ytest

def plot_training_data(Xtrain, Ytrain):
    image_number = random.randint(0, len(Xtrain) - 1)

    # Retrieve the file paths to the images
    X_image_path = Xtrain.iloc[image_number]['File_path']
    Y_image_path = Ytrain.iloc[image_number]['File_path']

    # Read the images using OpenCV
    X_image = cv2.imread(X_image_path)
    Y_image = cv2.imread(Y_image_path)

    # Convert BGR images to RGB for plotting with matplotlib
    X_image_rgb = cv2.cvtColor(X_image, cv2.COLOR_BGR2RGB)
    Y_image_rgb = cv2.cvtColor(Y_image, cv2.COLOR_BGR2RGB)

    # Plot the images
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.imshow(X_image_rgb)
    plt.title('Xtrain Image')

    plt.subplot(122)
    plt.imshow(Y_image_rgb)
    plt.title('Ytrain Image')

    plt.show()

# 3. Creation of new folders, necesary for ImageDataGenerator image detection
def createfolders(data_path,folder_names):
    for folder in folder_names:
        folder_path = os.path.join(data_path, folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print(f"Folder {folder} created at: {folder_path}")
        else: 
            print(f"Folder {folder} already exists at: {folder_path}")

    # Move images using shutil library
def move_images(df, folder_path):
    for _, i in df.iterrows():
        src = i['File_path']
        shutil.copy(src, folder_path)

# 4. Create subfolders for each label, inside of each folder
def images_class(df, folder_path):
    for _, i in df.iterrows():
        if i["Channel"] == "red":
            src = i['new_file_path']
            shutil.copy(src, folder_path)
        elif i["Channel"] == "green":
            src = i['new_file_path']
            shutil.copy(src, folder_path)
        else:
            src = i['new_file_path']
            shutil.copy(src, folder_path)
            
# 5. ImageDataGenerator --> Generates data for training and testing out of the split
            # Xtrain and Xtest: Microscopy images (czi stack)
            # Ytrain and Ytest: Mask images 
def get_generator(Xtrain_path, Xtest_path, Ytrain_path, Ytest_path): 
    img_data_gen_args = dict(rotation_range=90,
                        width_shift_range=0.3,
                        height_shift_range=0.3,
                        shear_range=0.5,
                        zoom_range=0.3,
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='reflect')

    mask_data_gen_args = dict(rotation_range=90,
                        width_shift_range=0.3,
                        height_shift_range=0.3,
                        shear_range=0.5,
                        zoom_range=0.3,
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='reflect',
                        preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype)) #Binarize the output again. 
    batch = 8 
    seed = 24
    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)

    # Images
    img_generator = image_data_generator.flow_from_directory(Xtrain_path, seed = seed, batch_size = batch,
                                                             target_size=(928, 928), shuffle= False, 
                                                             class_mode = None)
    valid_img_gen = image_data_generator.flow_from_directory(Xtest_path, seed = seed, batch_size = batch,
                                                             target_size=(928, 928), shuffle= False, 
                                                             class_mode = None)

    # Masks
    mask_generator = mask_data_generator.flow_from_directory(Ytrain_path, seed=seed, batch_size = batch,
                                                             target_size=(928, 928), shuffle= False, 
                                                             class_mode = None)
    valid_mask_gen = mask_data_generator.flow_from_directory(Ytest_path, seed=seed, batch_size = batch,
                                                             target_size=(928, 928), shuffle= False, 
                                                             class_mode = None)  #Default batch size 32, if not specified here

    return img_generator, valid_img_gen, mask_generator, valid_mask_gen

# 6. The above augmentation could have distortioned the masks values. Therefore,
    # we make a threshold so that the array of pixel values is between 0 and 1. 
# def threshold_mask(X_batch):
#     samples = X_batch.shape[0]
#     X_batch_new = np.zeros((X_batch.shape[0], X_batch.shape[1], X_batch.shape[2], 1), dtype=np.float64)

#     for i in range(samples):
#         temp = X_batch[i]
#         temp[temp <= 0.5] = 0.0
#         temp[temp > 0.5] = 1.0
#         temp = np.squeeze(temp, axis=-1)
#         X_batch_new[i] = temp

#     return X_batch_new

def threshold_mask(X_batch):
    samples = X_batch.shape[0]
    X_batch_new = np.zeros((X_batch.shape[0], X_batch.shape[1], X_batch.shape[2], 1), dtype=np.float64)

    for i in range(samples):
        temp = X_batch[i]
        temp[temp <= 0.5] = 0.0
        temp[temp > 0.5] = 1.0
        temp = np.expand_dims(temp[..., 0], axis=-1)  # Collapse the last dimension
        X_batch_new[i] = temp

    return X_batch_new


# 7. The U-net model only needs one generator. So image and mask generator must be combined. 
def combine_generators(image_generator, mask_generator):
    while True:
        yield (image_generator.next(), threshold_mask(mask_generator.next()))
