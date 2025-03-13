import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import argparse

# Delete a file
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

# Remove files containing a specified string
def remove_files_containing(data_path, string):
    
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    
    files_to_remove = [f for f in files if string in f]
    
    def delete_file(file_name):
        
        try:
            
            file_path = os.path.join(data_path, file_name)
            
            os.remove(file_path)
            
            print(f"Deleted: {file_name}")
            
        except Exception as e:
            
            print(f"Error deleting {file_name}: {e}")

    with ThreadPoolExecutor() as executor:
        
        executor.map(delete_file, files_to_remove)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Delete all files matching the filename in the given directory (recursive)")
    
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--filename', type=str, required=True)
    
    args = parser.parse_args()
    
    delete_files_in_directory(args.data_path, [args.filename])
