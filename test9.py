import cv2
import numpy as np
import os
from PIL import Image

# Directory containing the images
input_dir = 'D:/Docscn/image'  # Provide the correct path to your image directory
output_dir = 'D:/Docscn/Scnimg'
pdf_path = 'D:/Docscn/scanned_images.pdf'

# A4 size in pixels at 300 DPI
A4_WIDTH_PX = 2550
A4_HEIGHT_PX = 3300

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get a list of all files in the input directory
image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

if not image_files:
    print("No images found in the directory.")
else:
    processed_images = []

    for image_file in image_files:
        try:
            # Load the image
            image_path = os.path.join(input_dir, image_file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error loading image: {image_path}")
                continue

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

            # Append the processed image to the list
            processed_images.append(output_path)

        except Exception as e:
            print(f"An error occurred while processing {image_file}: {e}")

    # Convert the processed images to a PDF
    if processed_images:
        try:
            image_list = [Image.open(img).convert('RGB') for img in processed_images]

            # Resize each image to A4 size
            resized_images = []
            for img in image_list:
                # Maintain aspect ratio
                img.thumbnail((A4_WIDTH_PX, A4_HEIGHT_PX), Image.Resampling.LANCZOS)

                # Create a white background image with A4 size
                a4_image = Image.new('RGB', (A4_WIDTH_PX, A4_HEIGHT_PX), (255, 255, 255))

                # Calculate position to paste the resized image on the A4 page
                paste_position = ((A4_WIDTH_PX - img.width) // 2, (A4_HEIGHT_PX - img.height) // 2)

                # Paste the resized image onto the A4 background
                a4_image.paste(img, paste_position)

                # Append the A4-sized image to the list
                resized_images.append(a4_image)

            # Save the resized images as a PDF with 300 DPI
            resized_images[0].save(pdf_path, save_all=True, append_images=resized_images[1:], resolution=300)
            print(f"PDF saved successfully as {pdf_path}")

        except Exception as e:
            print(f"An error occurred while saving the PDF: {e}")
    else:
        print("No processed images to convert to PDF.")
