import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_uint
import PIL

# Paths to your images
fiber_image_path = r'C:\Users\DELL\Documents\2024\2024\evaporate_1uM_RhB_27032024\79_125\rhodamine_450_lampOn_200ms_550nmFEL.tiff'
background_image_path = r'C:\Users\DELL\Documents\2024\2024\evaporate_1uM_RhB_27032024\79_125\rhodamine_450_background_200ms_550nmFEL.tiff'

# Load the images
fiber_image = io.imread(fiber_image_path)
background_image = io.imread(background_image_path)

# Convert images to grayscale if necessary
if len(fiber_image.shape) > 2:
    fiber_image = color.rgb2gray(fiber_image)
if len(background_image.shape) > 2:
    background_image = color.rgb2gray(background_image)

# Perform background subtraction
subtracted_image = fiber_image.astype(float) - background_image.astype(float)

# Normalize the subtracted image to [0, 1]
subtracted_image = (subtracted_image - np.min(subtracted_image)) / (np.max(subtracted_image) - np.min(subtracted_image))

# Convert to uint16
subtracted_image = img_as_uint(subtracted_image)

# Convert to PIL image
subtracted_image_pil = PIL.Image.fromarray(subtracted_image)

# Set the DPI
subtracted_image_pil.info['dpi'] = (600, 600)

# Save the output image
output_path = r'C:\Users\DELL\Documents\2024\2024\evaporate_1uM_RhB_27032024\79_125\subtracted_image_normalized.tiff'
subtracted_image_pil.save(output_path)

# Display the result
plt.figure(figsize=(10, 5), dpi=600)
plt.subplot(1, 3, 1)
plt.imshow(fiber_image, cmap='gray')
plt.title('Fiber Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(background_image, cmap='gray')
plt.title('Background Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(subtracted_image, cmap='gray')
plt.title('Subtracted Image (Normalized)')
plt.axis('off')

plt.show()
