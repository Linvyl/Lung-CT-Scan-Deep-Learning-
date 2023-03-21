import cv2
import numpy as np

# Read the image
image = cv2.imread('ct_scan5.jpg', cv2.IMREAD_GRAYSCALE)

# Convert the image to HU format
slope, intercept = 0.05, -50
hu_image = slope * image.astype(np.float32) + intercept

# Apply Gaussian filter
gaussian = cv2.GaussianBlur(hu_image, (5, 5), 0)

# convert the gaussian image to 8-bit grayscale
gaussian_8bit = cv2.convertScaleAbs(gaussian)

# apply Otsu thresholding to the 8-bit grayscale image
_, binary = cv2.threshold(gaussian_8bit, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Select the contour corresponding to the lung area
lung_contour = None
max_area = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        lung_contour = contour

# Separate left and right lung contours
if lung_contour is not None:
    # Remove redundant dimension
    lung_contour = np.squeeze(lung_contour)
    left_lung_contour = lung_contour[lung_contour[:, 0].argmin():, :]
    right_lung_contour = lung_contour[:lung_contour[:, 0].argmax(), :]

    # Calculate convex hulls
    left_hull = cv2.convexHull(left_lung_contour)
    right_hull = cv2.convexHull(right_lung_contour)

    # Dilate and merge masks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    left_mask = cv2.dilate(cv2.fillConvexPoly(np.zeros_like(image), left_hull, 1), kernel)
    right_mask = cv2.dilate(cv2.fillConvexPoly(np.zeros_like(image), right_hull, 1), kernel)
    merged_mask = cv2.bitwise_or(left_mask, right_mask)

    # Apply mask to the original image
    masked_image = cv2.bitwise_and(hu_image, hu_image, mask=merged_mask)

    # Transform image to uint8
    masked_image = cv2.normalize(masked_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Crop image
    x, y, w, h = cv2.boundingRect(merged_mask)
    cropped_image = masked_image[y:y+h, x:x+w]

    # Display the images
    cv2.imshow('Original Image', image)
    cv2.imshow('HU Image', hu_image)
    cv2.imshow('Binary Image', binary)
    cv2.imshow('Left Lung Contour', cv2.drawContours(np.zeros_like(image), [left_lung_contour], 0, 255, 1))
    cv2.imshow('Right Lung Contour', cv2.drawContours(np.zeros_like(image), [right_lung_contour], 0, 255, 1))
    cv2.imshow('Merged Hull Mask', merged_mask)
    cv2.imshow('Masked Image', masked_image)
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)
