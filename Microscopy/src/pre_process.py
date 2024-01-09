import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import cv2
import shutil
import numpy as np
import pandas as pd
from keras.preprocessing import image
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
from keras.preprocessing.image import ImageDataGenerator 
from sklearn.model_selection import train_test_split

# Create dataframe containing the .tiff images of each folder. A dataframe for each folder, so the train and test split can be done. 
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
        # Cleaning df
        df["File_path1"] = df["File_path"]
        df["File_path1"] = df["File_path1"].str.extract(r'D:/TFM/Microscopy/video/copies/channels/(.*)')
        df["File_path1"] = df["File_path1"].str.replace("\\","/")
        df["Image_id"] = df["File_path1"].str.findall(r'(\d{4}-\d{1}[a-z]\d{4})').str[0]
        df[["Channel", "Copie"]] = df["File_path1"].str.split("/", expand=True)
        df[["Sample", "Duplicates","Timepoints"]] = df["Copie"].str.extract(r'(\d{4})-(\d{1})[a-z](\d{4})')
        df.drop(columns=["File_path1", "Copie"], inplace=True)

        lista = ["blue", "green", "red"]
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
        src = i['File_path']
        shutil.copy(src, folder_path)

# Create subfolders for each label, inside of each folder
def images_class(df, folder_path_blue, folder_path_green, folder_path_red):
    for _, i in df.iterrows():
        if i["Channel"] == "red":
            src = i['new_file_path']
            shutil.copy(src, folder_path_red)
        if i["Channel"] == "green":
            src = i['new_file_path']
            shutil.copy(src, folder_path_green)
        else:
            src = i['new_file_path']
            shutil.copy(src, folder_path_blue)

# Once we have the creation of the training and validation folders, and the creation of the rgb subfolders, we can start with the segmentation part
# For the segmentation part we need first to select the proposal regions to be the ones that we want, so we need to select them manually (manual annotation) with OpenCV

def threshold(img, thresh=127, mode='inverse'):
    im = img.copy()
    if mode == 'direct':
        thresh_mode = cv2.THRESH_BINARY
    else:
        thresh_mode = cv2.THRESH_BINARY_INV
     
    ret, thresh = cv2.threshold(im, thresh, 255, thresh_mode)
    return thresh

def display(img, thresh):
    display(img, thresh, 
        name_l='Original Image', 
        name_r='Thresholded Image',
        figsize=(20,14))

def get_bboxes(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort according to the area of contours in descending order.
    sorted_cnt = sorted(contours, key=cv2.contourArea, reverse = True)
    # Remove max area, outermost contour.
    sorted_cnt.remove(sorted_cnt[0])
    bboxes = []
    for cnt in sorted_cnt:
        x,y,w,h = cv2.boundingRect(cnt)
        cnt_area = w * h
        bboxes.append((x, y, x+w, y+h))
    return bboxes

def draw_annotations(img, bboxes, thickness=2, color=(0,255,0)):
    annotations = img.copy()
    for box in bboxes:
        tlc = (box[0], box[1])
        brc = (box[2], box[3])
        cv2.rectangle(annotations, tlc, brc, color, thickness, cv2.LINE_AA)
    return annotations

def morph_op(img, mode='open', ksize=5, iterations=1):
    im = img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize, ksize))
     
    if mode == 'open':
        morphed = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    elif mode == 'close':
        morphed = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    elif mode == 'erode':
        morphed = cv2.erode(im, kernel)
    else:
        morphed = cv2.dilate(im, kernel)
    return morphed

def get_filtered_bboxes(img, min_area_ratio=0.001):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort the contours according to area, larger to smaller.
    sorted_cnt = sorted(contours, key=cv2.contourArea, reverse = True)
    # Remove max area, outermost contour.
    sorted_cnt.remove(sorted_cnt[0])
    # Container to store filtered bboxes.
    bboxes = []
    # Image area.
    im_area = img.shape[0] * img.shape[1]
    for cnt in sorted_cnt:
        x,y,w,h = cv2.boundingRect(cnt)
        cnt_area = w * h
        # Remove very small detections.
        if cnt_area > min_area_ratio * im_area:
            bboxes.append((x, y, x+w, y+h))
    return bboxes


