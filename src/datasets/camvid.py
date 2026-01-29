import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import ConfusionMatrixDisplay
from PIL import Image
import os
from os import path,listdir,walk
from matplotlib import image
import numpy as np
import random
import numpy as np
import threading
import re
from concurrent.futures import ProcessPoolExecutor
from classification_models.tfkeras import Classifiers
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import cv2
import numpy as np

def get_y_fn(x,directory):
    filename=os.path.basename(x)
    split=os.path.splitext(filename)
    labelname=split[0]+'_L'+split[1]
    labelpath=path.join(directory,labelname)
    return labelpath

def sort_img (IMG_DIR,LAB_DIR):
    ids = listdir(IMG_DIR)
    labels=listdir(LAB_DIR)
    images_names = [path.join(IMG_DIR, image_id) for image_id in ids if path.isfile(path.join(IMG_DIR, image_id))]
    labels_names = [i for i in (get_y_fn(image_id,LAB_DIR) for image_id in os.listdir(IMG_DIR)) if path.isfile(i) ]
    return images_names,labels_names

def plot(images_names,labels_names, index):
    train_image = Image.open(images_names[index])
    labels= Image.open(labels_names[index])
    train_image.show()
    labels.show()
    return()

def open_image(fn,img_sz): 
    return np.array(Image.open(fn).resize(img_sz, Image.NEAREST))

def resize_image_with_aspect(sort_image_dir,sort_label_dir,factor):
    img_sz = [int(factor*960),int(factor*720)]
    imgs = np.stack([open_image(fn,img_sz) for fn in sort_image_dir])/255
    labels = np.stack([open_image(fn,img_sz) for fn in sort_label_dir])
    return imgs,labels

def parse_code(l):
    a,b = re.split('\\t+', l)  # - splits on one or more subsequent tabs, should they occur
    return tuple(int(o) for o in a.split(' ')), b[:-1]  # - [:-1] leaves out the newline at the end of each line

def conv_one_label(i): 
    res = np.zeros((r,c), 'uint8')
    for j in range(r): 
        for k in range(c):
            try: 
                res[j,k] = code2id[tuple(labels[i,j,k])]
            except: 
                res[j,k] = failed_code
    return res

def conv_all_labels():
    ex = ProcessPoolExecutor(8)
    return np.stack(ex.map(conv_one_label, range(n)))

def data_augmentation(input_image, output_image):
    # Data augmentation
    
    image_1 = cv2.flip(input_image, 1)
    mask_1 = cv2.flip(output_image, 1)
    
    image_2 = cv2.flip(input_image, 0)
    mask_2 = cv2.flip(output_image, 0)
    
    image_3 = cv2.flip(image_1, 0)
    mask_3 = cv2.flip(mask_1, 0)
    
    MASK = np.vstack([np.stack([mask_1]), np.stack([mask_2]), np.stack([mask_3])])
    IMAGE = np.vstack([np.stack([image_1]), np.stack([image_2]), np.stack([image_3])])

    return IMAGE, MASK
