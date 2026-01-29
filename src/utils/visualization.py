import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import math

import tensorflow_datasets as tfds


def plot_trainning_history(history):
    epochs=len(history["loss"])
    plt.figure(figsize=(12,9), dpi=60)

    for i, metric in enumerate(["accuracy", "loss"]):
        plt.subplot(1, 2, i+1)

        plt.plot(range(1,epochs+1), history[f'val_{metric}'],"r--", alpha=1.0, linewidth=2.0,label='validation')
        plt.plot(range(1,epochs+1), history[metric],"b--", alpha=1.0, linewidth=3.0,label='training')
        plt.legend(loc=0)
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.xlim([1,epochs+1])
        plt.grid(True)
        plt.title(f"Model {metric.capitalize()}")

    return plt


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
