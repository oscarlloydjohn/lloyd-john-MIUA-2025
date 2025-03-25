# Benny pointnet
import shutil

# Custom modules
from preprocess import *
from preprocessing_post_fastsurfer.subject import *
from preprocessing_post_fastsurfer.alignment import *
from preprocessing_post_fastsurfer.cropping import *
from preprocessing_post_fastsurfer.mesh_creation import *
from combined_models import *

def process_single_subject(subject: Subject):

    subject_data = {}

    # Get reference brain
    reference_brain_array = extract_brain(os.path.join(os.path.dirname(__file__), "mni_icbm152_lin_nifti/fastsurfer-processed/mri/orig_nu.mgz"), os.path.join(os.path.dirname(__file__), "mni_icbm152_lin_nifti/fastsurfer-processed/mri/mask.mgz"))

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
    mesh_dict = volume_to_mesh(subject, 'Left-Hippocampus_aligned_cropped.nii', smooth=True, number_of_iterations=5, lambda_filter=1.2)

    # Check there are enough points
    if len(mesh_dict['verts']) < 1048:

        print("Error: number of hippocampus mesh points is too low, the scan may be corrupted or inadequate")

    # Downsample the cloud to 1048 points for pointnet
    downsample_cloud(subject, "Left-Hippocampus_aligned_cropped_mesh.npz", 1048)
    
    # This is mimicking the torch Dataset
    
    # Load in the cloud
    subject_data['cloud'] = np.load(os.path.join(subject.path, 'Left-Hippocampus_aligned_cropped_mesh_downsampledcloud.npy'))

    # Get the volumetric data - this code is lifted from ozzy_torch_utils
    volume_col = subject.aseg_stats['Volume_mm3']

    volume_col_normalised = volume_col / volume_col.sum() * 1000

    struct_name_col = subject.aseg_stats['StructName']
    
    subject_data['volumes'] = np.array(volume_col_normalised)
    
    subject_data['struct_names'] = np.array(struct_name_col)

    return subject_data

def get_scores():

    score_names = ['MMSE Total Score', 'GDSCALE Total Score', 'FAQ Total Score', 'NPI-Q Total Score']

    def get_score_input(prompt, min_val, max_val):

        while True:

            try:

                value = input(prompt)

                if value == "":

                    return np.nan
                
                value = float(value)

                if min_val <= value <= max_val:

                    return value
                
                else:

                    print(f"Value must be between {min_val} and {max_val}.")

            except ValueError:

                print("Invalid input. Please enter a number or press enter for NaN.")

    mmse = get_score_input("Enter 'MMSE Total Score' (0-30), or press enter if no score: ", 0, 30)
    gdscale = get_score_input("Enter 'GDSCALE Total Score' (0-15), or press enter if no score: ", 0, 15)
    faq = get_score_input("Enter 'FAQ Total Score' (0-30), or press enter if no score: ", 0, 30)
    npiq = get_score_input("Enter 'NPI-Q Total Score' (0-12), or press enter if no score: ", 0, 12)

    scores = [mmse, gdscale, faq, npiq]

    return scores, score_names

# THIS WHOLE FILE NEEDS TO RUN IN A DOCKER CONTAINER
if __name__ == "__main__":

    '''parser = argparse.ArgumentParser(description="Script for running a prediction on a single .nii file")

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
    os.mkdir("/tmp/mripredict/")

    tmp_path = os.path.join("/tmp/mripredict/", filename)

    shutil.copy(args.file_path, tmp_path)

    # Run fastsurfer on file
    process_file("/tmp/mripredict/", filename, args.license_path, 4, tesla3=False)'''

    # Process the subject
    subject = Subject(os.path.join(os.path.dirname(__file__), "fastsurfer_sample"), None)

    subject_data = process_single_subject(subject)

    # Read in neurocognitive test scores
    if input("Does the subject have test scores? (y/n): ") == 'y':
        
        subject_data['scores'], subject_data['score_names'] = get_scores()

    else:

        subject_data['scores'], subject_data['score_names'] = None, None

    get_combined_prediction(subject)

    #shutil.rmtree("/tmp/mripredict/")