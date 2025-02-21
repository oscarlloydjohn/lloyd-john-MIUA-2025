import nibabel
import nibabel.affines
from PIL import Image
import os
import fnmatch
import numpy as np
import ants
import concurrent.futures
import pandas as pd
import glob
import xml.etree.ElementTree as ET
import xmltodict
import shutil
import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load an image into nibabel
def load_image(data_path, filename):
    
    return nibabel.load(f"{data_path}/{filename}")

# Display the middle slice of a nibabel image
def display_image(image):

    # Get image data array from image object
    image_array = np.asarray(image.dataobj)
    
    display_array(image_array)
    
    return 

# Display the middle slice of a 3d array
def display_array(array):
    
    # Get middle slice
    slice = array[array.shape[0] // 2, :, :]
    
    # Scale the image such that the maximum pixel value is 255
    # Display the scaled image
    display(Image.fromarray(((slice / np.max(slice)) * 255).astype(np.uint8)))
    
    return 

def display_image_3d(image):
    
    # Get image data array from image object
    image_array = np.asarray(image.dataobj)
    
    display_array_3d(image_array)
    
    return

def display_array_3d(array):
    
    array = (array / np.max(array)) * 255

    x, y, z = np.indices(array.shape)

    
    # Set up the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(x, y, z, array > 0, facecolors=plt.cm.viridis(array / 255.0), edgecolors='k')

    plt.show()
    
    return 

# Display the middle slice of a 3d array
def display_array(array):
    
    # Get middle slice
    slice = array[array.shape[0] // 2, :, :]
    
    # Scale the image such that the maximum pixel value is 255
    # Display the scaled image
    display(Image.fromarray(((slice / np.max(slice)) * 255).astype(np.uint8)))
    
    return 
