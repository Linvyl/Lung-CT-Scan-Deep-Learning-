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
image = cv2.imread('ct_scan5.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Normalize the grayscale image
normalized = normalize(gray)

# Apply the Otsu thresholding method to obtain a binary mask of the lung area
ret, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply morphological operations to eliminate below bolder areas
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# compute the convex hull
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
lung_hull = cv2.convexHull(contours[0])

# convert the single-channel image to a 3-channel image
lung_hull_3ch = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)

# draw the convex hull on the 3-channel image
cv2.drawContours(lung_hull_3ch, [lung_hull], -1, (0, 255, 0), 2)

# Dilate the lung hull
dilated = cv2.dilate(lung_hull_3ch, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)))

# Display the results
cv2.imshow('Original Image', image)
#cv2.imshow('Grayscale Image', gray)
#cv2.imshow('Normalized Image', normalized)
cv2.imshow('Binary Mask', binary)
cv2.imshow('Opened Mask', opened)
cv2.imshow('Convex Hull Mask', lung_hull_3ch)
cv2.imshow('Dilated Mask', dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()
