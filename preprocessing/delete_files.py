import os
import concurrent.futures

def delete_file(file_path):
    
    try:
        os.remove(file_path)
        
        print(f"Deleted: {file_path}")
        
    except Exception as e:
        
        print(f"Failed to delete {file_path}: {e}")

# Delete filenames matching list recursively
def delete_files_in_directory(data_path, filenames_to_delete):

    files_to_delete = []

    for dirpath, _, filenames in os.walk(data_path):
        
        for filename in filenames:
            
            if filename in filenames_to_delete:
                
                files_to_delete.append(os.path.join(dirpath, filename))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        
        executor.map(delete_file, files_to_delete)

if __name__ == "__main__":
    
    data_path = "/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/scratch disk/preprocessed-sample"
    
    filenames_to_delete = ["aparc.DKTatlas+aseg.deep.nii", "aseg.auto_noCCseg.nii", "mask.nii", "orig.nii", "orig_nu.nii"]
    
    delete_files_in_directory(data_path, filenames_to_delete)
