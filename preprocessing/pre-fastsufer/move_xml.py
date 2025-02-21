import os
import shutil

# Moves adni xml files to the appropriate directory based on matching the last 13 characters of the filename and directory name
def move_xml(directory):

    for dirpath, dirnames, filenames in os.walk(directory):

        for filename in filenames:
            
            if filename.lower().endswith('.xml'):

                base_filename = os.path.splitext(filename)[0]
                
                suffix = base_filename[-13:]

                for subdir in dirnames:
                    
                    if subdir.endswith(suffix):

                        source_file = os.path.join(dirpath, filename)
                        
                        target_directory = os.path.join(dirpath, subdir)

                        try:
                            
                            shutil.move(source_file, target_directory)
                            
                            print(f"Moved {filename} to {target_directory}")
                            
                        except Exception as e:
                            
                            print(f"Failed to move {filename} to {target_directory}: {e}")
                            
                        break

if __name__ == "__main__":
    
    data_path = "/vol/scratch/SoC/misc/2024/sc22olj/full-datasets/adni1-complete-3yr-3t"

    move_xml(data_path)
