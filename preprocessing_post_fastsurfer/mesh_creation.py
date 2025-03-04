import nibabel
import nibabel.affines
import os
import numpy as np
import concurrent.futures
from skimage import measure
import open3d as o3d

# Custom modules
from .vis import *

# Convert a volume of voxels to a pointcloud mesh using walking cubes
def volume_to_mesh(subject, fname_or_attribute):
    
    if os.path.exists(os.path.join(subject.path, fname_or_attribute)):
        
        image_array = nibabel.load(os.path.join(subject.path, fname_or_attribute)).get_fdata()
        
    else:
        
        image_array = nibabel.load(getattr(subject, fname_or_attribute)).get_fdata()
    
    verts, faces, normals, values = measure.marching_cubes(image_array)
    
    fname = os.path.splitext(os.path.basename(fname_or_attribute))[0]
    
    pathname = os.path.join(subject.path, fname + '_mesh.npz')
    
    np.savez(pathname, verts=verts, faces=faces, normals=normals, values=values)
    
    return {"verts" : verts, "faces" : faces, "normals" : normals, "values" : values}

def volume_to_mesh_parallel(subject_list, fname_or_attribute, downsample_factor=50, display=True):
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = []
        
        for subject in subject_list:

            futures.append(executor.submit(volume_to_mesh, subject, fname_or_attribute))
            
        for future in concurrent.futures.as_completed(futures):
            
            mesh = future.result()
            
            if display:
                
                display_mesh(mesh, downsample_factor)
    
    return
    
# Get the minimum number of vertices over all meshes in the list
def get_min_cloud_points(subject_list, filename):
    
    min_samples = np.inf
    
    for subject in subject_list:
        
        min_samples = min(min_samples, len(np.load(os.path.join(subject.path, filename), allow_pickle=True)['verts']))
    
    return min_samples

# https://medium.com/towards-data-science/how-to-use-pointnet-for-3d-computer-vision-in-an-industrial-context-3568ba37327e
# Farthest point sampling is suggested
# NB used open3d as it natively supports this type of sampling
def downsample_cloud(subject, filename, n):
    
    mesh_dict = np.load(os.path.join(subject.path, filename))
    
    cloud = o3d.geometry.PointCloud()
    
    cloud.points = o3d.utility.Vector3dVector(mesh_dict['verts'])

    downsampled_cloud = np.asarray(cloud.farthest_point_down_sample(n).points)
    
    basename = os.path.splitext(os.path.basename(filename))[0]
        
    downsampled_path = os.path.join(subject.path,(basename + '_downsampledcloud.npy'))
    
    np.save(downsampled_path, downsampled_cloud)
    
    return downsampled_cloud

def downsample_cloud_parallel(subject_list, filename, num_samples, display=True):
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = []
        
        for subject in subject_list:

            futures.append(executor.submit(downsample_cloud, subject, filename, num_samples))
            
        for future in concurrent.futures.as_completed(futures):
            
            cloud = future.result()
            
            if display:
                
                display_cloud(cloud)
    
    return
