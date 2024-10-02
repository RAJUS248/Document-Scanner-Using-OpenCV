import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('rs4.jpg')

# Preprocess the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Adjust the brightness and contrast
alpha = 1.5 # Contrast control (1.0-3.0)
beta = 0    # Brightness control (0-100)
adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(adjusted, (5, 5), 0)

# Apply Canny Edge Detection
edged = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Loop over the contours to find the document
doc_contour = None
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        doc_contour = approx
        break

# Apply perspective transform
def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

if doc_contour is not None:
    warped = four_point_transform(image, doc_contour.reshape(4, 2))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Apply adaptive threshold to get binary image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply bitwise_not to invert the image
    thresh = cv2.bitwise_not(thresh)

    # Display the result using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(thresh, cmap='gray')
    plt.title('Scanned Image')
    plt.axis('off')
    plt.show()
else:
    print("Document contour not found")
