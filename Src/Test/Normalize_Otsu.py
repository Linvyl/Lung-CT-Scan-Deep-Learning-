import cv2
import numpy as np

# Define the normalization function
def normalize(image):
    # Define the range of Hounsfield units (HU) corresponding to lung tissue
    lung_range = (-1000, 400)
    
    # Clip the image intensity values to the lung range
    image = np.clip(image, *lung_range)
    
    # Rescale the intensity values to the range [0, 1]
    image = (image - lung_range[0]) / (lung_range[1] - lung_range[0])
    
    # Convert the intensity range to the range [0, 255]
    image = (image * 255).astype(np.uint8)
    
    return image

# Read the CT image and convert it to grayscale
image = cv2.imread('ct_scan3.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Normalize the grayscale image
normalized = normalize(gray)

# Apply the Otsu thresholding method to obtain a binary mask of the lung area
ret, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply morphological operations to eliminate below bolder areas
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)



# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Grayscale Image', gray)
cv2.imshow('Normalized Image', normalized)
cv2.imshow('Binary Mask', binary)
cv2.imshow('Opened Mask', opened)
cv2.waitKey(0)
cv2.destroyAllWindows()
