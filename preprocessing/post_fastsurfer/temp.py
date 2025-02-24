import os
import pprint

# Example usage
dir1 = '/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/scratch disk/full-datasets/adni1-complete-1yr-3t'
dir2 = '/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/scratch disk/full-datasets/adni1-complete-3yr-3t'

dir1_contents = set(os.listdir(dir1))
dir2_contents = set(os.listdir(dir2))

# Find common names in both directories
common_names = dir1_contents.intersection(dir2_contents)

non_common_name = dir1_contents.symmetric_difference(dir2_contents)

pprint.pprint(len(common_names))

pprint.pprint(len(non_common_name))


# Iterate over common names and delete in the second directory (dir2)
for name in common_names:
    
    path_in_dir2 = os.path.join(dir2, name)
    
    if os.path.exists(path_in_dir2):
        try:
            # Check if it's a file or directory and remove accordingly
            if os.path.isdir(path_in_dir2):
                os.rmdir(path_in_dir2)  # Remove directory if empty
                print(f"Deleted directory: {path_in_dir2}")
            elif os.path.isfile(path_in_dir2):
                os.remove(path_in_dir2)  # Remove file
                print(f"Deleted file: {path_in_dir2}")
        except Exception as e:
            print(f"Error deleting {path_in_dir2}: {e}")