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

def show_images(processed_image):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=600)
    im1 = ax.imshow(processed_image, cmap='viridis')
    ax.set_title('Original Image')
    ax.axis('off')
    cbar1 = fig.colorbar(im1, ax=ax, orientation='vertical')
    cbar1.set_label('Intensity')
    plt.tight_layout()
    plt.show()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=600)
    zoom_center = (590, 570)
    zoom_radius = 200
    zoom_xmin = max(0, zoom_center[0] - zoom_radius)
    zoom_xmax = min(processed_image.shape[1], zoom_center[0] + zoom_radius)
    zoom_ymin = max(0, zoom_center[1] - zoom_radius)
    zoom_ymax = min(processed_image.shape[0], zoom_center[1] + zoom_radius)
    zoomed_image = processed_image[zoom_ymin:zoom_ymax, zoom_xmin:zoom_xmax]
    im2 = ax.imshow(zoomed_image, cmap='viridis')
    ax.set_title('Zoomed-in Image')
    ax.axis('off')
    cbar2 = fig.colorbar(im2, ax=ax, orientation='vertical')
    cbar2.set_label('Intensity')
    bar_line_position_x = 370
    bar_line_position_y = 360
    bar_thickness = 5
    conversion_factor = 120 / 690  # Adjust this based on your conversion factor
    bar_length_um = 10  # Length of the scale bar in µm
    bar_length_pixels = int(bar_length_um / conversion_factor)
    ax.plot([bar_line_position_x - bar_length_pixels, bar_line_position_x], [bar_line_position_y, bar_line_position_y], color='white', linewidth=bar_thickness)
    ax.text(bar_line_position_x - bar_length_pixels // 2, bar_line_position_y + 20, '10 µm', color='white', ha='center')

    plt.tight_layout()
    plt.show()

def main():
    fiber_image_path = r'C:\Users\DELL\Documents\2024\RhB_1uM_flow_from_water_fiber_22042024\all_100ms_1_min.tiff'
    fiber_image = load_image(fiber_image_path)
    processed_image = preprocess_image(fiber_image)
    center = (570, 590)
    diameter = 700
    processed_image_masked = create_circle_mask(processed_image, center, diameter)
    bar_line_position_x = 1100
    bar_line_position_y = 1100
    bar_thickness = 10
    conversion_factor = 120 / 690
    bar_length_um = 30
    text = '30 um'  # Use Unicode string
    processed_image_with_bar = draw_scale_bar(processed_image_masked, bar_line_position_x, bar_line_position_y, bar_thickness, conversion_factor, bar_length_um, text)
    brightness_factor = 0.9
    processed_image_brightened = adjust_brightness(processed_image_with_bar, brightness_factor)
    output_path = r'C:\Users\DELL\Documents\2024\RhB_1uM_flow_from_water_fiber_22042024\output_All.tiff'
    save_image(processed_image_brightened, output_path)
    show_images(processed_image_brightened)

if __name__ == "__main__":
    main()