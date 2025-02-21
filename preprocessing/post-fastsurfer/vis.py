import nibabel
import nibabel.affines
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display

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

def display_image_3d(image, downsample_factor):
    
    # Get image data array from image object
    image_array = np.asarray(image.dataobj)
    
    print("converted image to array")
    
    display_array_3d(image_array, downsample_factor)
    
    return

def display_array_3d(array, downsample_factor):
    
    array = array[::downsample_factor, ::downsample_factor, ::downsample_factor]
    
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')
    
    print('plotting')

    ax.voxels((array != 0), edgecolors='k')

    plt.show()
    
    return 

def display_mesh(verts, downsample_factor):
    
    num_samples = len(verts) // downsample_factor

    # Randomly sample the vertices (without replacement)
    sampled_indices = np.random.choice(len(verts), size=num_samples, replace=False)
    
    downsampled_verts = verts[sampled_indices]

    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(downsampled_verts[:, 0], downsampled_verts[:, 1], downsampled_verts[:, 2], c='b', s=1)  # Small point size
    
    plt.show()
    
    return
