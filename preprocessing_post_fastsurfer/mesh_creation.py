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
from skimage import measure

# Custom modules
from .vis import *

# Convert a volume of voxels to a pointcloud mesh using walking cubes
def volume_to_mesh(subject, fname_or_attribute):
    
    if os.path.exists(os.path.join(subject.path, fname_or_attribute)):
        
        image_array = nibabel.load(os.path.join(subject.path, fname_or_attribute)).get_fdata()
        
    else:
        
        image_array = nibabel.load(getattr(subject, fname_or_attribute)).get_fdata()
    
    verts, faces, _, _ = measure.marching_cubes(image_array)
    
    fname = os.path.splitext(os.path.basename(fname_or_attribute))[0]
    
    np.save(os.path.join(subject.path, fname + '_mesh.npy'), verts)
    
    return verts

def volume_to_mesh_parallel(subject_list, fname_or_attribute, downsample_factor=50):
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = []
        
        for subject in subject_list:

            futures.append(executor.submit(volume_to_mesh, subject, fname_or_attribute))
            
        for future in concurrent.futures.as_completed(futures):
            
            display_mesh(future.result(), downsample_factor)
    
    return

# Finds the mesh with the lowest number of points in the dataset
# Random samples all other meshes to match the number of points
def downsample(subject, filename, n):
    
        mesh = np.load(os.path.join(subject.path, filename))
        
        indices = np.random.choice(len(mesh), n, replace=False)
        
        downsampled_mesh = mesh[indices]
        
        basename = os.path.splitext(os.path.basename(filename))[0]
            
        downsampled_path = os.path.join(subject.path,(basename + '_downsampled.npy'))
        
        np.save(downsampled_path, downsampled_mesh)
        
        return downsampled_mesh
    
def get_samples(subject, filename):
    
    return len(np.load(os.path.join(subject.path, filename)))

def random_sample_mesh_parallel(subject_list, filename):
    
    min_samples = np.inf
    
    for subject in subject_list:
        
        min_samples = min(min_samples, get_samples(subject, filename))
        
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = []
        
        for subject in subject_list:

            futures.append(executor.submit(downsample, subject, filename, min_samples))
            
        for future in concurrent.futures.as_completed(futures):
            
            result = future.result()
            
            print(np.shape(result))
            
            print(len(result))
            
            display_mesh(result, 50)
    
    return
