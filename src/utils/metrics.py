import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import math

import tensorflow_datasets as tfds


from tqdm import tqdm
from collections import defaultdict
import tensorflow as tf
# labels.py was downloaded form https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
# The repo is by authors of cityscapes and provides helpful utilities
from labels import id2label, labels
import cv2

train_id_to_name = {}
train_id_to_color = {}

IGNORE_TRAIN_IDS =  (-1, 255)

for label in labels:
    if label.trainId not in IGNORE_TRAIN_IDS:
        train_id_to_name[label.trainId] = label.name
        train_id_to_color[label.trainId] = label.color

# Add void class
VOID_CLASS = len(train_id_to_color)
train_id_to_name[VOID_CLASS] = 'void'
train_id_to_color[VOID_CLASS] = (0, 0, 0)

def data_generator(ds, batch_size, resolution):
    resolution = (resolution[1], resolution[0]) # NUMPY CONVENTION :(
    for data in ds.batch(batch_size):
        label_batch = data["segmentation_label"].numpy()
        for label in labels:
            if label.trainId in IGNORE_TRAIN_IDS:
                label_batch[label_batch==label.id] = VOID_CLASS
            else:
                label_batch[label_batch==label.id] = label.trainId
        image = np.stack([cv2.resize(i.numpy(), resolution) for i in data["image_left"]])
        mask = np.stack([cv2.resize(i, resolution) for i in label_batch])
        yield image, mask


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def decode_segmentation_masks(mask, color_mapping):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    
    for class_id, class_color in color_mapping.items():
        idx = mask == class_id
        r[idx] = class_color[0]
        g[idx] = class_color[1]
        b[idx] = class_color[2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def visualize_np_array(array):
    from matplotlib import pyplot as plt
    plt.imshow(array, interpolation='nearest')
    plt.show()

def predict_and_visualize(model, testing_image):
    image, mask = testing_image
    predicted = np.argmax(model.predict(image), axis=3)[0]
    acutal = mask[0]
    print("Image Accuracy", np.sum(predicted==acutal)/np.size(acutal))

    visualize_np_array(image[0])
    visualize_np_array(decode_segmentation_masks(acutal, train_id_to_color))
    visualize_np_array(decode_segmentation_masks(predicted, train_id_to_color))



def calculate_metrics(model,test_ds, resolution):
    Y_actual = []
    Y_pred = []

    for data in tqdm(data_generator(test_ds, 4, resolution)):
        image, mask = data
        Y_pred.append( np.argmax(model.predict(image), axis=3))
        Y_actual.append(mask)

    Y_actual = np.concatenate(Y_actual)
    Y_pred = np.concatenate(Y_pred)

    print("Accuracy:", np.sum(Y_pred==Y_actual)/np.size(Y_actual))
    
    m = tf.keras.metrics.MeanIoU(num_classes=20)
    m.update_state(Y_actual, Y_pred)
    print("meanIoU", m.result().numpy())
    
    # Added later-on so doesn't appear when they are called during training
    cm = confusion_matrix(Y_actual.flatten(), Y_pred.flatten(), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(train_id_to_name.values()))
    fig, ax = plt.subplots(figsize=(20,20))
    disp.plot(ax=ax)
