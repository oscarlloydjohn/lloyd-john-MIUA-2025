import os

def find_files(directory, extension):
    """Recursively find all files with the given extension in the directory."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files

def get_basename(file_path):
    """Get the base name of the file without the extension."""
    return os.path.splitext(os.path.basename(file_path))[0]

def find_unmatched_files(dir1, dir2, ext1, ext2):
    """Find files in dir1 and dir2 with matching base names but different extensions."""
    files1 = find_files(dir1, ext1)
    files2 = find_files(dir2, ext2)
    
    basenames1 = {get_basename(f) for f in files1}
    basenames2 = {get_basename(f) for f in files2}
    
    unmatched_in_dir1 = basenames1 - basenames2
    unmatched_in_dir2 = basenames2 - basenames1
    
    return unmatched_in_dir1, unmatched_in_dir2

# Paths to the two directories
dir1 = "/vol/scratch/SoC/misc/2024/sc22olj/full-datasets/adni1-complete-1yr-3t/data"
dir2 = "/vol/scratch/SoC/misc/2024/sc22olj/full-datasets/adni1-complete-1yr-3t/metadata"

# Find unmatched files
unmatched_nii, unmatched_xml = find_unmatched_files(dir1, dir2, ".nii", ".xml")

# Print unmatched files
if unmatched_nii:
    print("Unmatched .nii files:")
    for basename in unmatched_nii:
        print(basename + ".nii")
else:
    print("All .nii files have matching .xml files.")

if unmatched_xml:
    print("Unmatched .xml files:")
    for basename in unmatched_xml:
        print(basename + ".xml")
else:
    print("All .xml files have matching .nii files.")