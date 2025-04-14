import shutil
import subprocess
import argparse

# Custom modules
from final_models_explainability.get_predictions import *
from pipeline_utils.image_processing import *
from pipeline_utils.get_scores import *
from pipeline_utils.frontend import *

"""
Prototype explainable AI pipeline for CN vs MCI prediction on neuroimaging data
======================================

A standalone script that demonstrates the outcomes of this project, including the prediction on an MRI volume and the explainability. The pipeline makes use of the processing functions from preprocessing_post_fastsurfer.

The script has two options for running, either run with no arguments to use a sample MRI /mri_samples/chris_t1 that has already been processed using fastsurfer. This skips the fastsurfer step but still performs the rest of the processing after that, then giving prediction and explainability.

To run with a custom MRI, pass in the --from_nii argument with the path to the .nii file. This will run Fastsurfer on the file before the rest of the pipeline, which takes about 15 minutes on CPU. The fastsurfer environment is built from a dockerfile, using docker or singularity (these must be installed). 

The script itself does not run inside a container due to the GUI, instead a lightweight requirements.txt file is provided called pipeline_requirements.txt.

Compatability and system requirements
---------
8Gb of ram is needed for fastsurfer inference, otherwise is pretty lightweight

Not tested on windows, probably doesn't work due to the way windows handles file paths.

For using the sample mri, ARM and x86 are supported 

For using custom MRI, only x86 architecure is supported which can be linux or macos. GPU is not enabled to maximise compatability as it is only working with 1 file anyway. ARM is not supported as fastsurfer provide no docker images for arm installs. It will actually run by emulating the architecture however it will probably take about a million years.

Arguments
---------

- **--from_nii** (*str*) **(required)**:
    The path to the .nii file to process. If not provided, a sample MRI will be used.

- **--threads** (*int*) **(required)**:  
    Number of threads to use for fastsurfer processing


Usage
-----

**Note, during the pipeline there may be popup windows that show some of the processing steps. These are from nibabel and are blocking calls, simply close the window to continue with the pipeline**

To set up the environment:
.. code-block:: bash

    python -m venv .pipelinevenv

    source .pipelinevenv/bin/activate

    pip install -r pipeline_requirements.txt


To run the script using the sample MRI:
.. code-block:: bash

    python3 pipeline.py

To run the script using a custom MRI. If using docker make sure it is running first by opening the docker app:
.. code-block:: bash

    python3 pipeline.py --from_nii <absolute path to .nii file> --threads <number of threads>

:author: Oscar Lloyd-John
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for running a prediction on a single .nii file")

    parser.add_argument('--from_nii', type=str, default='')
    parser.add_argument('--threads', type=int, default=4)

    args = parser.parse_args()

    # Make a copy in tmp for use in the pipeline
    os.makedirs("/tmp/mripredict/", exist_ok=True)

    # If a custom mri is specified
    if args.from_nii != '':

        if not str(args.from_nii).endswith('.nii') or not os.path.isfile(args.from_nii):
            print("Please pass in an nii file")
            exit()

        subject = run_fastsurfer(args.from_nii, args.threads)
    
    # If no custom mri, just use chris_t1. Also copies to tmp rather than working in that directory
    else:

        dirname = os.path.join(os.path.dirname(__file__), "mri_samples/chris_t1")
        tmp_path = os.path.join("/tmp/mripredict/", "chris_t1")
        shutil.copytree(dirname, tmp_path, dirs_exist_ok=True)

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