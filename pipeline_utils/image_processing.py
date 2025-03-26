import subprocess

# Custom modules
from preprocessing_post_fastsurfer.subject import *
from preprocessing_post_fastsurfer.alignment import *
from preprocessing_post_fastsurfer.cropping import *
from preprocessing_post_fastsurfer.mesh_creation import *

def process_single_subject(subject: Subject):

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

def run_fastsurfer(data_path: os.PathLike[str], filename: str, threads = 4):

    dirname = os.path.splitext(os.path.basename(filename))[0]

    command = [
        "/fastsurfer/run_fastsurfer.sh",
        f"--sd", data_path,
        "--sid", dirname,
        f"--t1", f"{data_path}/{filename}",
        "--seg_only", "--threads", f"{threads}"
    ]

    process = subprocess.Popen(command)
    
    process.wait()
    
    return