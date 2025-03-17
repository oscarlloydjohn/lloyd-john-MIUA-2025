import os
import fnmatch

from .subject import Subject

# Find files with a specific filename and return a list. Non-recursive 
def list_files_ext(data_path: os.PathLike[str], extensions: str) -> list:

    files = [f for f in os.listdir(data_path) if f.endswith(extensions)]
        
    return files       

# Return the absolute path to all files matching a filename in a directory. Recursive
def list_files_fname(data_path: os.PathLike[str], filename: str) -> list:
    
    matched_files = []
    
    for root, dirs, files in os.walk(data_path):
        
        for file in fnmatch.filter(files, filename):
            
            matched_files.append(os.path.join(root, file))
    
    return matched_files

# Delete the files returned by list_files_fname
def delete_file_matching(data_path: os.PathLike[str], filename: str) -> None:
    
    for file in list_files_fname(data_path, filename):
        
        os.remove(file)
    
    return

def delete_aux_files(subject: Subject) -> None:
    
    for file in subject.aux_file_list[:]:
        
            subject.aux_file_list.remove(file)
        
            os.remove(os.path.normpath(file))

    return