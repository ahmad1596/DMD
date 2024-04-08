import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, img_as_ubyte
import PIL
import cv2

def load_image(image_path):
    image = io.imread(image_path)
    return image

def preprocess_image(image):
    image = img_as_float(image)
    image -= np.min(image)
    image /= np.max(image)
    return np.clip(image, 0, 1)  

def adjust_brightness(image, brightness_factor):
    image_brightened = image * brightness_factor
    image_brightened = np.clip(image_brightened, 0, 1)  
    return image_brightened

def create_circle_mask(image, center, diameter):
    x, y = np.ogrid[:image.shape[0], :image.shape[1]]
    mask = ((x - center[0])**2 + (y - center[1])**2) <= (diameter / 2)**2
    image[~mask] = np.min(image)  
    return image

def draw_scale_bar(image, bar_line_position_x, bar_line_position_y, bar_thickness, conversion_factor, bar_length_um, text):
    bar_length_pixels = int(bar_length_um / conversion_factor)
    image[bar_line_position_y:bar_line_position_y + bar_thickness,
          bar_line_position_x - bar_length_pixels:bar_line_position_x] = 255
    text_size = 1
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_size, 2)[0][0]
    text_position_x = bar_line_position_x - bar_length_pixels + (bar_length_pixels - text_width) // 2
    text_position_y = bar_line_position_y + 50
    cv2.putText(image, text, (text_position_x, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def save_image(image, output_path):
    image_pil = PIL.Image.fromarray(img_as_ubyte(image))
    image_pil.info['dpi'] = (600, 600)
    image_pil.save(output_path)

def show_images(original_image, processed_image):
    plt.figure(figsize=(10, 5), dpi=600)
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Fiber Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(processed_image, cmap='gray')
    plt.title('Processed Fiber Image')
    plt.axis('off')
    plt.show()

def main():
    fiber_image_path = r'C:\Users\DELL\Documents\2024\100nM_BSA-TR_lamp550_04042024\600FEL_filter_1000ms_Core.tiff'

    fiber_image = load_image(fiber_image_path)

    processed_image = preprocess_image(fiber_image)

    center = (500, 735)
    diameter = 690
    processed_image_masked = create_circle_mask(processed_image, center, diameter)

    bar_line_position_x = 1100
    bar_line_position_y = 1100
    bar_thickness = 5
    conversion_factor = 120 / 690
    bar_length_um = 30
    text = '30 nm'
    processed_image_with_bar = draw_scale_bar(processed_image_masked, bar_line_position_x, bar_line_position_y, bar_thickness, conversion_factor, bar_length_um, text)

    brightness_factor = 0.6
    processed_image_brightened = adjust_brightness(processed_image_with_bar, brightness_factor)

    output_path = r'C:\Users\DELL\Documents\2024\100nM_BSA-TR_lamp550_04042024\output_Core.tiff'
    save_image(processed_image_brightened, output_path)

    show_images(fiber_image, processed_image_brightened)

if __name__ == "__main__":
    main()
