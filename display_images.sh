#!/bin/bash

folder_path="/home/debian/boot-scripts-master/boot-scripts-master/device/bone/capes/DLPDLCR2000/mask_outputs/"

duration=2

iterations=5

files=("$folder_path"/*.png)

if [ ${#files[@]} -eq 0 ]; then
  echo "No PNG files found in the specified folder."
  exit 1
fi

echo "Displaying PNG files in a loop..."
echo "Folder Path: $folder_path"
echo "Duration per Image: $duration seconds"
echo "Number of Iterations: $iterations (use '0' for infinite loop)"

for ((i=0; i<$iterations || $iterations==0; i++)); do
  for file in "${files[@]}"; do
    echo "Displaying: $file"
    feh -F "$file"
    sleep $duration
  done
done

echo "Display loop completed."

# Don't forget to make this script executable by running: chmod +x display_images.sh