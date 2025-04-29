Fastsurfer Processing Bash Script Instructions
=================================

This Bash script launches a Singularity container to run a FastSurfer preprocessing script preprocess.py on a dataset of NIfTI files. It allows GPU support and multithreading.

Environment Variables
----------------------

The script defines the following environment variables:

- **DATA_PATH**: Path to the dataset directory. Remember to change this by setting an environment variable or adjusting the script.

- **SCRIPT_PATH**: Path to the preprocessing Python script.  

- **LICENSE_PATH**: Path to the FreeSurfer license file, this must be obtained from freesurfer

- **CONTAINER_PATH**: Path to the Singularity image file (.sif), this is provided by FastSurfer

Command Breakdown
-----------------

The script runs this command:

.. code-block:: bash

    singularity exec --nv \
        --no-home \
        -B $DATA_PATH:$DATA_PATH \
        -B $LICENSE_PATH:$LICENSE_PATH \
        -B $SCRIPT_PATH:$SCRIPT_PATH \
        $CONTAINER_PATH \
        python3 $SCRIPT_PATH --threads 12 --data_path $DATA_PATH --license_path $LICENSE_PATH

Example Usage
-------------

To run the script, make sure it's executable:

.. code-block:: bash

    chmod +x fastsurfer_pipeline.sh

Then run:

.. code-block:: bash

    ./fastsurfer_pipeline.sh

To run this overnight on a lab machine, use:

.. code-block:: bash

    nohup ./fastsurfer_pipeline.sh > [log_path] 2>&1 & disown

This will run the script in the background and save the output to a log file. To kill it you can use ps aux or top to get the process ID.