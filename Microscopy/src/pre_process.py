import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import re
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator 
from sklearn.model_selection import train_test_split

# Create dataframe containing the .tiff images of each folder. A dataframe for each folder, so the train and test split can be done. 
# dfs[0] is blue, dfs[1] is green, dfs[2] is red

def create_df(folder):
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    dfs = []

    for sub in subfolders:
        list_tiff = []

        for file in os.listdir(sub):
            list_tiff.append(os.path.join(sub, file))

        df = pd.DataFrame(list_tiff, columns=["File_path"])
        # Cleaning df
        df["File_path1"] = df["File_path"]
        print(df)
        df["File_path1"] = df["File_path1"].str.extract(r'D:/TFM/Microscopy/video/copies/(.*)')
        df[["Channel", "Copie"]] = df["File_path1"].str.split("\\", expand=True)
        df[["Sample", "Duplicates", "Timepoints"]] = df["Copie"].str.extract(r'(\d{4})-(\d{1})copia(\d{4})')
        df.drop(columns=["File_path1", "Copie"], inplace=True)

        lista = ["blue", "green", "red"]
        for color in lista:
            df.loc[df['Channel'] == color, 'Label'] = lista.index(color)

        dfs.append(df)

    blue, green, red = dfs[0], dfs[1], dfs[2]
    # Concatenate df vertically. pd.merge() is not an option as it combines df horizontally.
    semi = pd.concat([blue, green], axis=0)
    all_df = pd.concat([red, semi], axis=0)

    return blue, green, red, all_df


# Split data into train and validation sets
def train_valid_split(df):
    train = df.drop(columns= ["Label"])
    valid = df["Label"]
    train_x, valid_x, train_y, valid_y = train_test_split(train,valid, test_size=0.2, stratify=df["Channel"], random_state=50)
    print("The proportion of train x data is:", train_x['Channel'].value_counts() / len(train_x.Channel))
    print("The proportion of valid x data is:", valid_x['Channel'].value_counts() / len(valid_x.Channel))
    return train_x, valid_x, train_y, valid_y

# Creation of new folders, necesary for ImageDataGenerator image detection
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
        src = i['file_path']
        shutil.move(src, folder_path)

# Create subfolders for each label, inside of each folder
def images_class(df, folder_path_blue, folder_path_green, folder_path_red):
    for _, i in df.iterrows():
        if i["Channel"] == "blue":
            src = i['new_file_path']
            shutil.move(src, folder_path_blue)
        if i["Channel"] == "green":
            src = i['new_file_path']
            shutil.move(src, folder_path_green)
        else:
            src = i['new_file_path']
            shutil.move(src, folder_path_red)


# Tried use mean subtraction, normalization, and standards to scale pixels, 
# however each of these methods affected the colors of the images. Found an 
# alternate approach in the "ImageDataGenerator" function. 

# An image generator for each CNN. Tried to join this 
# function using conditionals but it was giving errors constantly. 

def image_generator():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        "D:/bootcamp/original/try_train/",  # this is the target directory
        target_size=(500, 500),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='binary',
        shuffle=False)  
    
    validation_generator = test_datagen.flow_from_directory(
        "D:/bootcamp/original/try_val/",
        target_size=(500, 500),
        batch_size=32,
        class_mode='binary',
        shuffle=False)
    
    return train_generator, validation_generator

def plot_augmented_images(train_generator, num_images=5):
    original_images = next(train_generator)
    original_image = original_images[0]

    original_image = np.expand_dims(original_image, axis=0)
    augmented_iterator = train_generator
    augmented_images = [next(augmented_iterator)[0][0].astype(np.uint8) for _ in range(num_images)]

    plt.figure(figsize=(15, 5))

    for i, augmented_image in enumerate(augmented_images):
        plt.subplot(1, num_images + 1, i + 2)
        plt.imshow(augmented_image)
        plt.title(f'Augmented {i + 1}')

    plt.show()