import os
import shutil

# Recursively search all subdirectories for files and put them in data_path. Delete empty directories
def compact_dir(data_path):
    
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(data_path, topdown=False):
        
        for filename in files:
            
            file_path = os.path.join(root, filename)
            
            # Move the file to the top-level directory
            shutil.move(file_path, os.path.join(data_path, filename))
        
        # After moving files, check and remove empty directories
        for dirname in dirs:
            
            dir_path = os.path.join(root, dirname)
            
            if not os.listdir(dir_path):  # Directory is empty
                
                os.rmdir(dir_path)
    
    print(f"Compaction of directory {data_path} complete.")