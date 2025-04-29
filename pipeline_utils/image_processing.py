"""
Image procesing
===========

This module provides functions processing a single MRI scan for use in the pipeline. It uses functions from preprocessing_post_fastsurfer, and also can run fastsurfer.

:author: Oscar Lloyd-John
"""

import subprocess
import argparse
import os
import shutil

# Custom modules
from preprocessing_post_fastsurfer.subject import *
from preprocessing_post_fastsurfer.alignment import *
from preprocessing_post_fastsurfer.cropping import *
from preprocessing_post_fastsurfer.mesh_creation import *


def process_single_subject(subject: Subject) -> dict:
    """

    Process a single subject's MRI scan using preprocessing_post_fastsurfer. Stores the image data in a dict similar to the ozzy_torch_utils dataset, meaning it is in memory.

    :param subject: The Subject object initialised after fastsurfer processing
    :type subject: Subject
    :return: A dict containing all the subject data
    :rtype: dict
    """

    subject_data = {}

    # Get reference brain
    reference_brain_array = extract_brain(os.path.join(os.path.dirname(__file__), "mni_orig_nu.mgz"), os.path.join(os.path.dirname(__file__), "mni_mask.mgz"))

    # Align the MRI
    print("Aligning subject brain with template brain \n")
    alignment(subject, reference_brain_array)

    # Align the aparc
    print("Aligning parcellation file \n")
    aux_alignment(subject, subject.aparc, is_aparc=True)

    # Extract the aligned left hippocampus
    print("Aligning left hippocampus \n")
    extract_region(subject, [17], subject.brain_aligned, subject.aparc_aligned, is_aligned=True)

    # Get the bounding box of the hippocampus
    print("Cropping hippocampus \n")
    hcampus_image_array = nibabel.load(os.path.join(subject.path, "Left-Hippocampus_aligned.nii")).get_fdata()

    bbox = bounding_box(hcampus_image_array)

    # Crop the left hippocampus
    crop(subject, "Left-Hippocampus_aligned.nii", bbox)

    # Convert the left hippocampus to mesh
    print("Converting hippocampus volume to mesh \n")
    mesh_dict = volume_to_mesh(subject, 'Left-Hippocampus_aligned_cropped.nii', smooth=True, number_of_iterations=5, lambda_filter=1.2)

    # Check there are enough points
    if len(mesh_dict['verts']) < 1048:

        print("Error: number of hippocampus mesh points is too low, the scan may be corrupted or inadequate")

    # Downsample the cloud to 1048 points for pointnet
    print("Sampling vertices from hippocampus mesh \n")
    downsample_cloud(subject, "Left-Hippocampus_aligned_cropped_mesh.npz", 1048)
    
    # This is mimicking the torch Dataset
    
    # Load in the cloud
    subject_data['lhcampus_pointcloud_aligned'] = np.load(os.path.join(subject.path, 'Left-Hippocampus_aligned_cropped_mesh_downsampledcloud.npy'))

    # Get the volumetric data - this code is lifted from ozzy_torch_utils
    print("Calculating parcellation region volume ratios \n")
    volume_col = subject.aseg_stats['Volume_mm3']

    volume_col_normalised = volume_col / volume_col.sum() * 1000

    struct_name_col = subject.aseg_stats['StructName']
    
    subject_data['volumes'] = np.array(volume_col_normalised)
    
    subject_data['struct_names'] = np.array(struct_name_col)

    return subject_data

def run_fastsurfer(path, threads) -> Subject:

    # Check if Singularity or Docker is installed
    singularity_installed = shutil.which("singularity") is not None
    docker_installed = shutil.which("docker") is not None

    if not singularity_installed and not docker_installed:
        print("Neither singularity or docker are installed, try using the sample MRI by running without --from_nii")
        exit(1)
    singularity_image = "fastsurfer-cpu.sif"

    if singularity_installed:
        if not os.path.isfile(singularity_image):
            print(f"Singularity image {singularity_image} not found, building it now")
            try:
                subprocess.run(
                    [
                        "singularity", "build", singularity_image,
                        "docker://deepmi/fastsurfer:cpu-v2.4.2"
                    ],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"Error building Singularity image: {e}")
                exit(1)

    filename = os.path.basename(path)
    tmp_path = os.path.join("/tmp/mripredict/", filename)
    shutil.copy(path, tmp_path)

    # Run FastSurfer using Singularity or Docker
    if singularity_installed:
        print("Running Fastsurfer using singularity")
        singularity_command = [
            "singularity", "exec",
            "--no-home",
            "-B", "/tmp/mripredict:/tmp/mripredict",
            singularity_image,
            "/fastsurfer/run_fastsurfer.sh",
            "--sd", "/tmp/mripredict",
            "--sid", os.path.splitext(filename)[0],
            "--t1", f"/tmp/mripredict/{filename}",
            "--seg_only",
            "--threads", f"{threads}"
        ]
        try:
            subprocess.run(singularity_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Fastsurfer with singularity: {e}")
            exit(1)

    elif docker_installed:
        print("Running Fastsurfer using docker")
        docker_command = docker_command = [
            "docker", "run", "--rm",
            "--user", "0",
            "-v", "/tmp/mripredict:/tmp/mripredict",
            "deepmi/fastsurfer:cpu-v2.4.2",
            "--t1", f"/tmp/mripredict/{filename}",
            "--sid", os.path.splitext(filename)[0],
            "--sd", "/tmp/mripredict",
            "--seg_only",
            "--threads", f"{threads}",
            "--allow_root"
        ]
        try:
            subprocess.run(docker_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Fastsurfer with docker: {e}")
            exit(1)

    subject = Subject(f"/tmp/mripredict/{os.path.splitext(filename)[0]}", None)

    return subject