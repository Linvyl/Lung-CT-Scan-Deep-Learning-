import cv2
import numpy as np

# Define the normalization function
def normalize(image):
    # Convert the image to Hounsfield units (HU)
    intercept = -1024
    slope = 1
    image = image.astype(np.float32) * slope + intercept
    
    # Clip the image intensity values to the lung range
    lung_range = (-1000, 400)
    image = np.clip(image, *lung_range)
    
    # Rescale the intensity values to the range [0, 1]
    image = (image - lung_range[0]) / (lung_range[1] - lung_range[0])
    
    # Convert the intensity range to the range [0, 255]
    image = (image * 255).astype(np.uint8)
    
    return image

# Read the CT image and convert it to grayscale
image = cv2.imread('ct_scan5.jpg', cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Normalize the grayscale image
normalized = normalize(gray)

# Apply the Otsu thresholding method to obtain a binary mask of the lung area
ret, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Select the connected domain of lung volume
components = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S)
component_sizes = components[2][:, cv2.CC_STAT_AREA]
lung_label = np.argmax(component_sizes[1:]) + 1
lung_mask = (components[1] == lung_label).astype(np.uint8)

# Separate the left and right lungs
left_mask = np.zeros_like(lung_mask)
right_mask = np.zeros_like(lung_mask)
h, w = lung_mask.shape[:2]
left_mask[:, :w // 2] = lung_mask[:, :w // 2]
right_mask[:, w // 2:] = lung_mask[:, w // 2:]

# Calculate the convex hull on each of lung side
left_contours, _ = cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
left_hull = cv2.convexHull(left_contours[0])
right_contours, _ = cv2.findContours(right_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
right_hull = cv2.convexHull(right_contours[0])

# Dilate and merge the two masks
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)).astype(np.uint8)
dilated_left_hull = cv2.dilate(left_hull.astype(np.uint8), kernel)
dilated_right_hull = cv2.dilate(right_hull.astype(np.uint8), kernel)

# Resize the masks to have the same dimensions
h, w = dilated_left_hull.shape[:2]
dilated_right_hull = cv2.resize(dilated_right_hull, (w, h))

# Merge the two masks
merged_hull = cv2.bitwise_or(dilated_left_hull, dilated_right_hull)

# Multiply the raw image with the mask
merged_hull = merged_hull.astype(np.uint8) # Ensure the mask is of type uint8
masked_image = cv2.bitwise_and(image, image, mask=merged_hull)

# Fill the masked area with tissue luminance
masked_image[np.where(merged_hull == 0)] = 160

# Convert the image to UINT8
masked_image = masked_image.astype(np.uint8)

# Crop the image into a specific size
cropped_image = masked_image[150:1100, 150:1100]

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Normalized Image', normalized)
cv2.imshow('Binary Mask', binary)
cv2.imshow('Merged Hull', merged_hull)
cv2.imshow('Masked Image', masked_image)
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()