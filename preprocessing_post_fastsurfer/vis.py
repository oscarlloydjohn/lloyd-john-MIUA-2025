import nibabel
import nibabel.affines
from PIL import Image
import numpy as np
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

def display_image_3d(image, downsample_factor, mode='interactive'):
    
    # Get image data array from image object
    image_array = np.asarray(image.dataobj)
    
    print("converted image to array")
    
    display_array_3d(image_array, downsample_factor, mode)
    
    return


def display_array_3d(array, downsample_factor, mode='interactive'):
    
    if mode == 'preview':
        
        '''plotter = pv.ImageData(window_size=[100, 100], notebook=False)'''
        
    elif mode == 'interactive':
        
        grid = pv.ImageData()
        
    array = array[::downsample_factor, ::downsample_factor, ::downsample_factor]
    
    grid.dimensions = np.array(array.shape) + 1
    
    grid.cell_data["values"] = array.flatten(order="F")
    
    grid_thresh = grid.threshold(1)
    
    grid_thresh.plot()
    
    return 

def display_cloud(cloud, mode='interactive'):

    mesh = pv.PolyData(cloud)

    if mode == 'preview':
        
        '''plotter = pv.ImageData(window_size=[100, 100], notebook=False)
        
        plotter.enable_anti_aliasing(False)'''
        
    elif mode == 'interactive':
        
        plotter = pv.Plotter()
    
    plotter.add_mesh(mesh, color='blue', point_size=5, render_points_as_spheres=True)
    
    plotter.set_background("white")
    
    plotter.show()
    
    return

def display_mesh(mesh_dict, mode='interactive'):
    
    if mode == 'preview':
        
        plotter = pv.Plotter(window_size=[100, 100], notebook=False)
        
    elif mode == 'interactive':
        
        plotter = pv.Plotter()
        
    faces = mesh_dict['faces']
    
    # Convert to correct shape for pyvista
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    
    mesh = pv.PolyData(mesh_dict['verts'], faces_pv)
    
    plotter.add_mesh(mesh, color="lightblue", show_edges=True)
    
    plotter.set_background("white")

    plotter.show()
        
    return

'''
def display_array_3d(array, downsample_factor, mode='interactive'):
    
    if mode == 'preview':
        plotter = pv.Plotter(off_screen=True, window_size=[100, 100])
        
    elif mode == 'interactive':
        plotter = pv.Plotter()
    
    # Downsample the array
    array = array[::downsample_factor, ::downsample_factor, ::downsample_factor]
    
    # Create ImageData object
    image_data = pv.ImageData(dimensions=np.array(array.shape) + 1)
    image_data.cell_data["values"] = array.flatten(order="F")
    
    # Apply threshold
    grid_thresh = image_data.threshold(1)
    
    plotter.add_mesh(grid_thresh, show_edges=True)
    plotter.set_background("white")
    plotter.show()
    
    return

def display_cloud(cloud, mode='interactive'):

    mesh = pv.PolyData(cloud)

    if mode == 'preview':
        plotter = pv.Plotter(off_screen=True, window_size=[100, 100])
        plotter.enable_anti_aliasing(False)
        
    elif mode == 'interactive':
        plotter = pv.Plotter()
    
    plotter.add_mesh(mesh, color='blue', point_size=5, render_points_as_spheres=True)
    plotter.set_background("white")
    plotter.show()
    
    return

def display_mesh(mesh_dict, mode='interactive'):
    
    if mode == 'preview':
        plotter = pv.Plotter(off_screen=True, window_size=[100, 100])
        plotter.enable_anti_aliasing(False)
        
    elif mode == 'interactive':
        plotter = pv.Plotter()
    
    faces = mesh_dict['faces']
    
    # Convert faces to the correct format for PyVista
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    
    # Create a PolyData mesh from vertices and faces
    mesh = pv.PolyData(mesh_dict['verts'], faces_pv)
    
    plotter.add_mesh(mesh, color="lightblue", show_edges=True)
    plotter.set_background("white")
    plotter.show()
    
    return'''
