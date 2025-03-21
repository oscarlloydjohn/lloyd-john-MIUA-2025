
# Torch
import torch
import torch.optim as optim
from torcheval.metrics import *
import torch.nn.functional as F

# Benny pointnet
from pointnet2_benny import pointnet2_cls_ssg
import shutil

# Custom modules
from preprocessing_pre_fastsufer.preprocess import *
from preprocessing_post_fastsurfer.subject import *
from preprocessing_post_fastsurfer.alignment import *
from preprocessing_post_fastsurfer.vis import *
from preprocessing_post_fastsurfer.cropping import *
from preprocessing_post_fastsurfer.mesh_creation import *
from ..explain_pointnet import *
from ozzy_torch_utils.split_dataset import *
from ozzy_torch_utils.subject_dataset import *
from ozzy_torch_utils.plot import *
from ozzy_torch_utils.train_nn import *
from ozzy_torch_utils.model_parameters import *
from ozzy_torch_utils.init_dataloaders import *

# THIS WHOLE FILE NEEDS TO RUN IN A DOCKER CONTAINER
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for running a prediction on a single .nii file")

    parser.add_argument('--tesla3', action='store_true')
    parser.add_argument('--file_path', type=str, required=True)
    parser.add_argument('--license_path', type=str, required=True)

    args = parser.parse_args()

    filename = os.path.basename(args.file_path)

    if not os.path.exists(args.file_path):
        
        print(f"{args.file_path} does not exist")
        
        exit()

    if not str(args.file_path).endswith('.nii'):

        print("Please pass in an nii file")
        
        exit()

    # Make a copy in tmp for use in the pipeline
    tmp_path = os.path.join("/tmp/mripredict/", filename)

    shutil.copy(args.file_path, tmp_path)

    # Load in nii image
    process_file("/tmp/mripredict/", filename, args.license_path, 4, args.tesla3)

    subject = Subject(tmp_path, None)

    # Get reference brain
    reference_brain_array = extract_brain("/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/mni_icbm152_lin_nifti/fastsurfer-processed/mri/orig_nu.mgz", "/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/mni_icbm152_lin_nifti/fastsurfer-processed/mri/mask.mgz")

    # Align the MRI
    alignment(subject, reference_brain_array)

    # Align the aparc
    aux_alignment(subject, subject.aparc, is_aparc=True)

    # Extract the aligned left hippocampus
    extract_region(subject, [17], subject.brain_aligned, subject.aparc_aligned, is_aligned=True)

    # Get the bounding box of the hippocampus
    hcampus_image_array = nibabel.load(os.path.join(subject.path, "Left-Hippocampus_aligned.nii")).get_fdata()

    bbox = bounding_box(hcampus_image_array)

    # Crop the left hippocampus
    crop(subject, "Left-Hippocampus_aligned.nii", bbox)

    # Convert the left hippocampus to mesh
    volume_to_mesh(subject, 'Left-Hippocampus_aligned_cropped.nii', smooth=True, number_of_iterations=5, lambda_filter=1.2)

    # Check there are enough points

    # Downsample the cloud to 1048 points for pointnet
    downsample_cloud(subject, "Left-Hippocampus_aligned_cropped_mesh.npz")

    shutil.rmtree("/tmp/mripredict/")