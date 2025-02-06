import os
import subprocess
import concurrent.futures

def find_files_recursively(directory, extension):
    """Recursively find all files with the given extension in the directory."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files

def process_file(file, license_path, container_path):
    """Run a subprocess on the given file."""
    data_path = os.path.dirname(file)
    output_path = data_path  # Set output path to the same directory as the input file
    t1_path = f"/data/{os.path.relpath(file, data_path)}"  # Set t1_path relative to the container's binding
    sid = os.path.basename(file)[5:15]  # Extract sid from the 10 characters after the first 5 characters of the filename
    command = [
        "singularity", "exec", "--nv",
        "--no-home",
        f"-B {data_path}:/data",
        f"-B {output_path}:/output",
        f"-B {license_path}:/freesurfer-license",
        container_path,
        "/fastsurfer/run_fastsurfer.sh",
        "--fs_license", "/freesurfer-license/license.txt",
        "--sd", "/output",
        "--sid", sid,
        "--t1", t1_path,
        "--3T", "--seg_only"
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in process.stdout:
        print(line, end='')
    for line in process.stderr:
        print(line, end='')
    process.wait()

    return

# Example usage
license_path = "/vol/scratch/SoC/misc/2024/sc22olj/tools/freesurfer-license"
container_path = "/vol/scratch/SoC/misc/2024/sc22olj/fastsurfer-gpu.sif"

#print(find_files_recursively("/vol/scratch/SoC/misc/2024/sc22olj/test-dataset-2-processed/adni-collection", ".nii"))

#process_file('/vol/scratch/SoC/misc/2024/sc22olj/test-dataset-2-processed/adni-collection/005_S_0572/MPR-R__GradWarp__B1_Correction__N3__Scaled/2006-12-27_14_37_18.0/I79141/ADNI_005_S_0572_MR_MPR-R__GradWarp__B1_Correction__N3__Scaled_Br_20071028192123921_S24527_I79141.nii' , license_path, container_path)



def run_in_parallel(directory, extension):
    """Find all files with the given extension and process them in parallel."""
    files = find_files_recursively(directory, extension)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, file, license_path, container_path): file for file in files}
        for future in concurrent.futures.as_completed(futures):
            file = futures[future]
            try:
                future.result()
                print(f"Successfully processed {file}")
            except Exception as exc:
                print(f"File {file} generated an exception: {exc}")

    return

# Path to the directory containing .nii files
directory = "/vol/scratch/SoC/misc/2024/sc22olj/test-dataset-2-processed/adni-collection"

# Run the subprocess in parallel on all .nii files
run_in_parallel(directory, ".nii")