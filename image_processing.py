import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
import PIL
import cv2

def load_images(fiber_image_path, background_image_path):
    fiber_image = io.imread(fiber_image_path)
    background_image = io.imread(background_image_path)
    return fiber_image, background_image

def preprocess_images(fiber_image, background_image):
    fiber_image = img_as_float(fiber_image)
    background_image = img_as_float(background_image)
    subtracted_image = fiber_image - background_image
    subtracted_image -= np.min(subtracted_image)
    subtracted_image /= np.max(subtracted_image)
    return subtracted_image

def adjust_brightness(subtracted_image, brightness_factor):
    subtracted_image_darkened = subtracted_image * brightness_factor
    return subtracted_image_darkened

def create_circle_mask(subtracted_image_darkened, center, diameter):
    x, y = np.ogrid[:subtracted_image_darkened.shape[0], :subtracted_image_darkened.shape[1]]
    mask = ((x - center[0])**2 + (y - center[1])**2) <= (diameter / 2)**2
    subtracted_image_darkened_copy = subtracted_image_darkened.copy()
    subtracted_image_darkened_copy[~mask] = 0
    return subtracted_image_darkened_copy

def draw_scale_bar(subtracted_image_darkened, bar_line_position_x, bar_line_position_y, bar_thickness, conversion_factor, bar_length_um, text):
    bar_length_pixels = int(bar_length_um / conversion_factor)
    subtracted_image_darkened_copy = subtracted_image_darkened.copy()
    subtracted_image_darkened_copy[bar_line_position_y:bar_line_position_y + bar_thickness,
                              bar_line_position_x - bar_length_pixels:bar_line_position_x] = 1
    text_size = 1
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_size, 2)[0][0]
    text_position_x = bar_line_position_x - bar_length_pixels + (bar_length_pixels - text_width) // 2
    text_position_y = bar_line_position_y + 50
    cv2.putText(subtracted_image_darkened_copy, text, (text_position_x, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (1, 1, 1), 2, cv2.LINE_AA)
    return subtracted_image_darkened_copy

def save_image(subtracted_image_darkened, output_path):
    subtracted_image_pil = PIL.Image.fromarray(subtracted_image_darkened)
    subtracted_image_pil.info['dpi'] = (600, 600)
    subtracted_image_pil.save(output_path)

def show_images(fiber_image, background_image, subtracted_image_darkened):
    plt.figure(figsize=(15, 5), dpi=100)

    plt.subplot(1, 3, 1)
    fiber_plot = plt.imshow(fiber_image, cmap='viridis') 
    plt.title('Fiber Image')
    plt.axis('off')
    plt.colorbar(fiber_plot, ax=plt.gca())

    plt.subplot(1, 3, 2)
    background_plot = plt.imshow(background_image, cmap='viridis') 
    plt.title('Background Image')
    plt.axis('off')
    plt.colorbar(background_plot, ax=plt.gca())

    plt.subplot(1, 3, 3)
    subtracted_plot = plt.imshow(subtracted_image_darkened, cmap='viridis', vmin=0, vmax=1)  
    plt.axis('off')
    plt.colorbar(subtracted_plot, ax=plt.gca())

    plt.tight_layout()
    plt.show()

def main():
    fiber_image_path = r'C:\Users\DELL\Documents\2024\100nM_BSA-TR_lamp550_04042024\600FEL_filter_1000ms_All.tiff'
    background_image_path = r'C:\Users\DELL\Documents\2024\100nM_BSA-TR_lamp550_04042024\background_600FEL_filter_1000ms.tiff'

    fiber_image, background_image = load_images(fiber_image_path, background_image_path)

    subtracted_image = preprocess_images(fiber_image, background_image)
    
    brightness_factor = 0.75
    subtracted_image_darkened = adjust_brightness(subtracted_image, brightness_factor)
    
    center = (500, 740)
    diameter = 690
    subtracted_image_darkened = create_circle_mask(subtracted_image_darkened, center, diameter)
        
    bar_line_position_x = 1100
    bar_line_position_y = 1100
    bar_thickness = 5
    conversion_factor = 120 / 690
    bar_length_um = 30
    text = '30 nm'
    subtracted_image_darkened = draw_scale_bar(subtracted_image_darkened, bar_line_position_x, bar_line_position_y, bar_thickness, conversion_factor, bar_length_um, text)

    output_path = r'C:\Users\DELL\Documents\2024\100nM_BSA-TR_lamp550_04042024\output_All.tiff'
    save_image(subtracted_image_darkened, output_path)

    show_images(fiber_image, background_image, subtracted_image_darkened)

if __name__ == "__main__":
    main()
