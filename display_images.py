#!/usr/bin/env python

import os
import time
from glob import glob

# Set the path to the folder containing the PNG files
folder_path = "/home/debian/boot-scripts-master/boot-scripts-master/device/bone/capes/DLPDLCR2000/mask_outputs/"

duration = 2

iterations = 5

files = glob(os.path.join(folder_path, '*.png'))

if not files:
    print("No PNG files found in the specified folder.")
    exit(1)

print("Displaying PNG files in a loop...")
print(f"Folder Path: {folder_path}")
print(f"Duration per Image: {duration} seconds")
print(f"Number of Iterations: {iterations} (use '0' for infinite loop)")

for i in range(iterations) if iterations > 0 else iter(int, 1):
    for file in files:
        print(f"Displaying: {file}")
        os.system(f"feh -F {file}")
        time.sleep(duration)

print("Display loop completed.")
