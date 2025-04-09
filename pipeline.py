import shutil
import subprocess
import argparse

# Custom modules
from final_models_explainability.get_predictions import *
from pipeline_utils.image_processing import *
from pipeline_utils.get_scores import *
from pipeline_utils.frontend import *

"""
Prototype explainable AI pipeline for neuroimaging data
======================================

A standalone script that demonstrates the outcomes of this project, including the prediction on an MRI volume and the explainability. The pipeline makes use of the processing functions from preprocessing_post_fastsurfer.

The script has two options for running, either run with no arguments to use a sample MRI /mri_samples/chris_t1 that has already been processed using fastsurfer. This skips the fastsurfer step but still performs the rest of the processing after that, then giving prediction and explainability.

To run with a custom MRI, pass in the --from_nii argument with the path to the .nii file. This will run Fastsurfer on the file before the rest of the pipeline. The fastsurfer environment is built from a dockerfile, using docker or singularity (these must be installed). Currently, only x86 architecure is supported. Mac ARM users will need to use the sample MRI

The script itself does not run inside a container due to the GUI, instead a lightweight requirements.txt file is provided called pipeline_requirements.txt.


Arguments
---------

- **--from_nii** (*str*) **(required)**:
    The path to the .nii file to process. If not provided, a sample MRI will be used.

- **--threads** (*int*) **(required)**:  
    Number of threads to use for fastsurfer processing


Usage
-----

To set up the environment:
.. code-block:: bash

    python -m venv .pipelinevenv

    source .pipelinevenv/bin/activate

    pip install -r pipeline_requirements.txt


To run the script using the sample MRI:
.. code-block:: bash

    python3 pipeline.py

To run the script using a custom MRI:
.. code-block:: bash

    python3 pipeline.py --from_nii <absolute path to .nii file> --threads <number of threads>

:author: Oscar Lloyd-John
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for running a prediction on a single .nii file")

    parser.add_argument('--from_nii', type=str, default='')
    parser.add_argument('--threads', type=int, default=4)

    args = parser.parse_args()

    if args.from_nii != '':

        if not str(args.from_nii).endswith('.nii') or not os.path.isfile(args.from_nii):
            print("Please pass in an nii file")
            exit()

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

    # Make a copy in tmp for use in the pipeline
    os.makedirs("/tmp/mripredict/", exist_ok=True)

    if args.from_nii != '':

        filename = os.path.basename(args.from_nii)
        tmp_path = os.path.join("/tmp/mripredict/", filename)
        shutil.copy(args.from_nii, tmp_path)

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
                "--threads", args.threads
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
                "--threads", args.threads,
                "--allow_root"
            ]
            try:
                subprocess.run(docker_command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running Fastsurfer with docker: {e}")
                exit(1)

        subject = Subject("/tmp/mripredict/{filename}", None)
    
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