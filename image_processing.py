import os
import cv2
import numpy as np
import shutil

def enhance_sharpness(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

def prepare_input_files(input_dir, source_dir, rhodamine_file, background_file, num_copies):
    os.makedirs(input_dir, exist_ok=True)
    
    background_dst = os.path.join(input_dir, 'background_1.png')
    shutil.copy(os.path.join(source_dir, background_file), background_dst)
    
    for i in range(1, num_copies + 1):
        rhodamine_dst = os.path.join(input_dir, f'rhodamine_{i}.png')
        shutil.copy(os.path.join(source_dir, rhodamine_file), rhodamine_dst)

def process_colored_images(input_dir, output_dir, input_prefix, output_prefix, num_images, integration_time=1):
    os.makedirs(output_dir, exist_ok=True)
    stacked_images = []
    first_image_path = os.path.join(input_dir, f'{input_prefix}_1.png')
    first_image = cv2.imread(first_image_path)
    height, width, channels = first_image.shape
    video_output_path = os.path.join(output_dir, f'{output_prefix}_video.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, 1/integration_time, (width, height))
    
    background_path = os.path.join(input_dir, 'background_1.png')
    background_img = cv2.imread(background_path)
    
    for i in range(1, num_images + 1):
        fibers_path = os.path.join(input_dir, f'rhodamine_{i}.png')
        fibers_img = cv2.imread(fibers_path)
        
        subtracted_img = cv2.subtract(fibers_img, background_img)
        subtracted_img_filtered = cv2.medianBlur(subtracted_img, 5)
        subtracted_img_filtered = enhance_sharpness(subtracted_img_filtered)
        
        output_subtracted_path = os.path.join(output_dir, f'{output_prefix}_subtracted_{i}.png')
        cv2.imwrite(output_subtracted_path, subtracted_img_filtered)
        video_writer.write(subtracted_img_filtered)
        stacked_images.append(subtracted_img_filtered)
    
    video_writer.release()
    stacked_images = np.stack(stacked_images, axis=-1)
    stacked_output_path = os.path.join(output_dir, f'{output_prefix}_stacked.png')
    cv2.imwrite(stacked_output_path, stacked_images)

def main():
    source_dir = r'C:\Users\DELL\Documents\2024'
    input_dir = r'C:\Users\DELL\Documents\temp'
    output_dir = os.path.join(input_dir, 'filtered_images')
    input_prefix = 'background'
    output_prefix = 'background_filtered'
    num_images = 20
    integration_time = 1
    num_copies = 20

    rhodamine_file = 'rhodamine_file_name.png' 
    background_file = 'background_file_name.png'  

    prepare_input_files(input_dir, source_dir, rhodamine_file, background_file, num_copies)
    process_colored_images(input_dir, output_dir, input_prefix, output_prefix, num_images, integration_time)

if __name__ == "__main__":
    main()
