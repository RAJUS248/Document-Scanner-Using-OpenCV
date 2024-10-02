import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('rs2.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding to binarize the image
_, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Invert the image to have a white background and black text
thresh_otsu = cv2.bitwise_not(thresh_otsu)

# Apply denoising
denoised = cv2.fastNlMeansDenoising(thresh_otsu, None, 30, 7, 21)

# Invert the image back to have black text on white background
denoised = cv2.bitwise_not(denoised)

# Save the scanned image
cv2.imwrite('scanned_image.jpg', denoised)

# Display the result using matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(denoised, cmap='gray')
plt.title('Scanned Image')
plt.axis('off')
plt.show()
