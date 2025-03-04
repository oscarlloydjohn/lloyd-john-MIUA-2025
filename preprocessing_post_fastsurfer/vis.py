import nibabel
import nibabel.affines
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display
import pyvista as pv

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
    
    grid = pv.ImageData()
    
    grid.dimensions = np.array(array.shape) + 1
    
    grid.cell_data["values"] = array.flatten(order="F")
    
    grid_thresh = grid.threshold(1)
    
    grid_thresh.plot()
    
    return 

def display_cloud(cloud):

    mesh = pv.PolyData(cloud)

    plotter = pv.Plotter()
    
    plotter.add_mesh(cloud, color="lightblue", show_edges=True)
    
    plotter.set_background("white")
    
    plotter.show()

def display_mesh(mesh_dict, downsample_factor, mode='mpl'):

    plotter = pv.Plotter()

    mesh = pv.PolyData(mesh_dict['verts'], mesh_dict['faces'])
    
    plotter.add_mesh(mesh, color="lightblue", show_edges=True)
    
    plotter.set_background("white")
    
    plotter.show()
        
    return
