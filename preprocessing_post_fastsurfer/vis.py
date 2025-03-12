import nibabel
import nibabel.affines
from PIL import Image
import numpy as np
from IPython.display import display
import pyvista as pv
import matplotlib.pyplot as plt

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

def display_image_3d(image, downsample_factor, mode='interactive'):
    
    # Get image data array from image object
    image_array = np.asarray(image.dataobj)
    
    print("converted image to array")
    
    display_array_3d(image_array, downsample_factor, mode)
    
    return


def display_array_3d(array, downsample_factor, mode='interactive'):
    
    # Preview mode is not implemented yet
    if mode == 'preview':
        
        '''plotter = pv.ImageData(window_size=[100, 100], notebook=False)'''
        
    elif mode == 'interactive':
        
        pass
    
    grid = pv.ImageData()
        
    array = array[::downsample_factor, ::downsample_factor, ::downsample_factor]
    
    grid.dimensions = np.array(array.shape) + 1
    
    grid.cell_data["values"] = array.flatten(order="F")
    
    grid_thresh = grid.threshold(1)
    
    grid_thresh.plot()
    
    return 

def display_cloud(cloud, mode='interactive'):

    mesh = pv.PolyData(cloud)
    
    if mode == 'mpl':

        fig = plt.figure()
        
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c='b', s=1)  # Small point size
        
        plt.show()
        
        return

    # Preview mode is not implemented yet
    if mode == 'preview':
        
        '''plotter = pv.ImageData(window_size=[100, 100], notebook=False)'''
        
    elif mode == 'interactive':
        
        pass
        
    plotter = pv.Plotter()
    
    plotter.add_mesh(mesh, color='blue', point_size=5, render_points_as_spheres=True)
    
    plotter.set_background("white")
    
    plotter.show()
    
    return

def display_mesh(mesh_dict, mode='interactive'):
    
    # Preview mode is not implemented yet
    if mode == 'preview':
        
        '''plotter = pv.ImageData(window_size=[100, 100], notebook=False)'''
        
    elif mode == 'interactive':
        
        pass
        
    plotter = pv.Plotter()
        
    faces = mesh_dict['faces']
    
    # Convert to correct shape for pyvista
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    
    mesh = pv.PolyData(mesh_dict['verts'], faces_pv)
    
    plotter.add_mesh(mesh, color="lightblue", show_edges=True)
    
    plotter.set_background("white")

    plotter.show()
        
    return
