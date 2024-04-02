import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_uint
import PIL

fiber_image_path = r'C:\Users\DELL\Documents\2024\600FEL_filter_200ms.tiff'
background_image_path = r'C:\Users\DELL\Documents\2024\100nM_BSA-TR_lamp550_28032024\background_600FEL_filter_200ms.tiff'

fiber_image = io.imread(fiber_image_path)
background_image = io.imread(background_image_path)

if len(fiber_image.shape) > 2:
    fiber_image = color.rgb2gray(fiber_image)
if len(background_image.shape) > 2:
    background_image = color.rgb2gray(background_image)

subtracted_image = fiber_image.astype(float) - background_image.astype(float)
subtracted_image = (subtracted_image - np.min(subtracted_image)) / (np.max(subtracted_image) - np.min(subtracted_image))
subtracted_image = img_as_uint(subtracted_image)
subtracted_image_pil = PIL.Image.fromarray(subtracted_image)
subtracted_image_pil.info['dpi'] = (600, 600)
output_path = r'C:\Users\DELL\Documents\2024\100nM_BSA-TR_lamp550_28032024\output.tiff'
subtracted_image_pil.save(output_path)

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
