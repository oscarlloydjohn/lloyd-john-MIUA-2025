#!/bin/bash

# Directories to compare
dir1="$1"
dir2="$2"

# List of filenames and directory names in dir1 and dir2 (excluding subdirectories)
find "$dir1" -maxdepth 1 -type f -exec basename {} \; > dir1_files_and_dirs.txt
find "$dir1" -maxdepth 1 -type d -exec basename {} \; | grep -v "^$" >> dir1_files_and_dirs.txt

find "$dir2" -maxdepth 1 -type f -exec basename {} \; > dir2_files_and_dirs.txt
find "$dir2" -maxdepth 1 -type d -exec basename {} \; | grep -v "^$" >> dir2_files_and_dirs.txt

# Output filenames and directories that are unique to dir1 (not in dir2)
echo "Unique to $dir1:"
comm -23 <(sort dir1_files_and_dirs.txt) <(sort dir2_files_and_dirs.txt)

# Output filenames and directories that are unique to dir2 (not in dir1)
echo -e "\nUnique to $dir2:"
comm -13 <(sort dir1_files_and_dirs.txt) <(sort dir2_files_and_dirs.txt)

# Clean up temporary files
rm dir1_files_and_dirs.txt dir2_files_and_dirs.txt
