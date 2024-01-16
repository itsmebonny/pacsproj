#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <folder_path>"
    exit 1
fi

# Assign the input folder path to a variable
folder_path="$1"

# Check if the provided path is a directory
if [ ! -d "$folder_path" ]; then
    echo "Error: '$folder_path' is not a valid directory."
    exit 1
fi

# Iterate over all files in the folder and run dolfin-convert
for file in "$folder_path"/*.msh; do
    if [ -f "$file" ]; then
        # Extract the filename without extension
        filename=$(basename "$file" .msh)
        
        # Run dolfin-convert
        echo "Converting file: $file"
        dolfin-convert "$file" "$folder_path/$filename.xml"
    fi
done

echo "Conversion complete."
