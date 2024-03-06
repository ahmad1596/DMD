import os
import cv2
import numpy as np

def enhance_sharpness(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

def process_colored_images(input_dir, output_dir, input_prefix, output_prefix, num_images, integration_time=1):
    os.makedirs(output_dir, exist_ok=True)
    sum_images = None
    first_image_path = os.path.join(input_dir, f'{input_prefix}_1.png')
    first_image = cv2.imread(first_image_path)
    height, width, channels = first_image.shape
    print(f"\nImage Width: {width}, Height: {height}")
    video_output_path = os.path.join(output_dir, f'{output_prefix}_video.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, 1/integration_time, (width, height))
    for i in range(1, num_images + 1):
        background_path = os.path.join(input_dir, f'{input_prefix}_{i}.png')
        fibers_path = os.path.join(input_dir, f'rhodamine_{i}.png')
        background_img = cv2.imread(background_path)
        fibers_img = cv2.imread(fibers_path)
        subtracted_img = cv2.subtract(fibers_img, background_img)
        subtracted_img_filtered = cv2.medianBlur(subtracted_img, 5)
        subtracted_img_filtered = enhance_sharpness(subtracted_img_filtered)
        output_subtracted_path = os.path.join(output_dir, f'{output_prefix}_subtracted_{i}.png')
        cv2.imwrite(output_subtracted_path, subtracted_img_filtered)
        video_writer.write(subtracted_img_filtered)
        if sum_images is None:
            sum_images = subtracted_img_filtered.astype(np.float32)
        else:
            sum_images += subtracted_img_filtered.astype(np.float32)
    video_writer.release()
    average_img = (sum_images / num_images).astype(np.uint8)
    average_output_path = os.path.join(output_dir, f'{output_prefix}_average.png')
    cv2.imwrite(average_output_path, average_img)

def main():
    input_dir = r'C:\Users\DELL\Documents\temp'
    output_dir = os.path.join(input_dir, 'filtered_images')
    input_prefix = 'background'
    output_prefix = 'background_filtered'
    num_images = 20
    integration_time = 1
    process_colored_images(input_dir, output_dir, input_prefix, output_prefix, num_images, integration_time)
if __name__ == "__main__":
    main()
