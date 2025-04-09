import shutil
import shap
import argparse

# Custom modules
from final_models_explainability.get_predictions import *
from pipeline_utils.image_processing import *
from pipeline_utils.get_scores import *
from pipeline_utils.frontend import *

# THIS WHOLE FILE NEEDS TO RUN IN A DOCKER CONTAINER FOR FROM_NII
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for running a prediction on a single .nii file")

    parser.add_argument('--from_nii', type=str, default='')

    args = parser.parse_args()

    if args.from_nii != '':

        if not str(args.from_nii).endswith('.nii') or not os.path.isfile(args.from_nii):

            print("Please pass in an nii file")
            
            exit()

    # Make a copy in tmp for use in the pipeline
    os.makedirs("/tmp/mripredict/", exist_ok=True)

    if args.from_nii != '':

        filename = os.path.basename(args.from_nii)

        tmp_path = os.path.join("/tmp/mripredict/", filename)

        shutil.copy(args.from_nii, tmp_path)

        # Run fastsurfer on file
        run_fastsurfer("/tmp/mripredict/", filename, args.license_path, 4)

        subject = Subject("/tmp/mripredict", None)
    
    else:

        dirname = os.path.join(os.path.dirname(__file__), "mri_samples/chris_t1")

        tmp_path = os.path.join("/tmp/mripredict/", "chris_t1")

        shutil.copytree(dirname, tmp_path, dirs_exist_ok=True)

        # Process the subject
        subject = Subject(tmp_path, None)

    subject_data = process_single_subject(subject)

    # Read in neurocognitive test scores
    get_scores(subject_data)

    # Get individual predictions and explainability
    print("Running inference on hippocampus pointcloud \n")
    pointnet_pred_class, pointnet_output, attributions = get_pointnet_prediction(subject_data['lhcampus_pointcloud_aligned'], 'cpu')

    print("Running inference on brain parcellation volumes \n")
    volumes_pred_class, volumes_output, shap_values = get_volumes_prediction(subject_data['volumes'], subject_data['struct_names'])

    # Get ensemble predictions
    if subject_data['scores'] is not None:

        print("Running inference on test scores \n")
        scores_pred_class, scores_output = get_scores_prediction(subject_data['scores'])

        print("Calculating ensemble prediction \n")
        prediction = get_ensemble_prediction_avg(pointnet_output, volumes_output, scores_output, scores=True)

    else:

        print("Calculating ensemble prediction \n")
        prediction = get_ensemble_prediction_avg(pointnet_output, volumes_output, None, scores=False)

    
    norm_xyz_sum = normalise_attributions(attributions)

    hcampus_plotter = vis_attributions(subject_data['lhcampus_pointcloud_aligned'], norm_xyz_sum)

    volumes_plotter = vis_volumes(subject, shap_values)

    print("Opening main window \n")

    show_main_window(prediction, hcampus_plotter, volumes_plotter, shap_values)

    print(prediction)

    shutil.rmtree("/tmp/mripredict/")