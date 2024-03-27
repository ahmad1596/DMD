import os
import cv2
import numpy as np

source_dir = r'C:\Users\DELL\Documents\2024'
output_dir = r'C:\Users\DELL\Documents\optofluidics-master\optofluidics-master\Python'
output_folder_name = 'rhodamine_subtracted_images'
rhodamine_file = 'rhodamine_450lamp_emission_5000ms_lampoff_afterflow_550FEL_fixed.tiff'
background_file = 'rhodamine_450lamp_emission_5000ms_lampoff_afterflow_550FEL_background_fixed.tiff'
output_file = 'stacked_subtracted_image.jpg'
num_copies = 10
rhodamine_img = cv2.imread(os.path.join(source_dir, rhodamine_file), cv2.IMREAD_UNCHANGED)
background_img = cv2.imread(os.path.join(source_dir, background_file), cv2.IMREAD_UNCHANGED)
os.makedirs(output_dir, exist_ok=True)
output_subtracted_dir = os.path.join(output_dir, output_folder_name)
os.makedirs(output_subtracted_dir, exist_ok=True)
subtracted_images = []
for i in range(num_copies):
    rhodamine_copy = rhodamine_img.copy()
    subtracted_img = cv2.subtract(rhodamine_copy, background_img)
    output_subtracted_file = f'subtracted_image_{i}.jpg'
    cv2.imwrite(os.path.join(output_subtracted_dir, output_subtracted_file), subtracted_img)
    subtracted_images.append(subtracted_img)
stacked_subtracted_images = np.stack(subtracted_images)
output_stacked_path = os.path.join(output_subtracted_dir, output_file)
cv2.imwrite(output_stacked_path, stacked_subtracted_images[0])  
