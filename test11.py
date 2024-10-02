import cv2
import numpy as np
import os
from PIL import Image

# Directories and file paths
input_dir = 'D:/Docscn/image'  # Directory containing the images
pdf_path = 'D:/Docscn/scannedimages.pdf'

# A4 size in pixels at 300 DPI
A4_WIDTH_PX = 2550
A4_HEIGHT_PX = 3300

def process_image(image):
    """Process the image: convert to grayscale, binarize, denoise."""
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

    return denoised

def save_images_as_pdf(image_paths, pdf_path):
    """Save the list of images as a single PDF with A4 dimensions."""
    resized_images = []
    for image_path in image_paths:
        # Process the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        processed_image = process_image(image)

        # Convert the processed image to PIL format
        pil_image = Image.fromarray(processed_image)
        pil_image = pil_image.convert('RGB')

        # Resize to A4 size, maintaining aspect ratio
        pil_image.thumbnail((A4_WIDTH_PX, A4_HEIGHT_PX), Image.Resampling.LANCZOS)

        # Create a white background image with A4 size
        a4_image = Image.new('RGB', (A4_WIDTH_PX, A4_HEIGHT_PX), (255, 255, 255))

        # Calculate position to paste the resized image on the A4 page
        paste_position = ((A4_WIDTH_PX - pil_image.width) // 2, (A4_HEIGHT_PX - pil_image.height) // 2)

        # Paste the resized image onto the A4 background
        a4_image.paste(pil_image, paste_position)

        # Append the A4-sized image to the list
        resized_images.append(a4_image)

    # Save the A4-sized images as a PDF
    if resized_images:
        resized_images[0].save(pdf_path, save_all=True, append_images=resized_images[1:], resolution=300)
        print(f"PDF saved successfully as {pdf_path}")
    else:
        print("No images to save as PDF.")

if __name__ == "__main__":
    # Get a list of all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    image_files = [os.path.join(input_dir, f) for f in image_files]

    if not image_files:
        print("No images found in the directory.")
    else:
        save_images_as_pdf(image_files, pdf_path)
