#!/bin/bash

DATA_PATH="/vol/scratch/SoC/misc/2024/sc22olj/full-datasets/hcampus-large-cohort/5"
SCRIPT_PATH="/uolstore/home/users/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/preprocessing_pre_fastsufer/preprocess.py"
LICENSE_PATH="/vol/scratch/SoC/misc/2024/sc22olj/tools/freesurfer-license/license.txt"
CONTAINER_PATH="/vol/scratch/SoC/misc/2024/sc22olj/fastsurfer-gpu.sif"

singularity exec --nv \
    --no-home \
    -B $DATA_PATH:$DATA_PATH \
    -B $LICENSE_PATH:$LICENSE_PATH \
    -B $SCRIPT_PATH:$SCRIPT_PATH \
    $CONTAINER_PATH \
    python3 $SCRIPT_PATH --process --data_path $DATA_PATH --license_path $LICENSE_PATH --container_path $CONTAINER_PATH
