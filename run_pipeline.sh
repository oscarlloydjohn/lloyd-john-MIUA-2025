# Check if Docker is installed
if command -v docker &> /dev/null; then
    
    docker pull deepmi/fastsurfer:cpu-v2.4.2


# If Docker is not available, check for Singularity/Apptainer
elif command -v singularity &> /dev/null || command -v apptainer &> /dev/null; then

    singularity build fastsurfer-cpu.sif docker://deepmi/fastsurfer:cpu-v2.4.2

    singularity exec \
    --no-home \
    -B $DATA_PATH:$DATA_PATH \

    /fastsurfer/run_fastsurfer.sh --sd data_path --sid dirname \
    --t1 data_path/filename --seg_only --threads num_threads

else
    echo "Cannot run with argument --from_nii, neither Docker nor Singularity/Apptainer found. Try running with the example nii"
    exit 1
fi
