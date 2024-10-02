import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Directory containing the images
input_dir = 'image'
output_dir = 'Scnimg'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get a list of all files in the input directory
image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

for image_file in image_files:
    # Load the image
    image_path = os.path.join(input_dir, image_file)
    image = cv2.imread(image_path)

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
    output_path = os.path.join(output_dir, 'scanned_' + image_file)
    cv2.imwrite(output_path, denoised)

    # Display the result using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(denoised, cmap='gray')
    plt.title('Scanned Image')
    plt.axis('off')
    plt.show()
