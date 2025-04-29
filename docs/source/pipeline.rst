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