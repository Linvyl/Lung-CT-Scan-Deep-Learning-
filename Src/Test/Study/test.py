import cv2
import numpy as np

# Load the JPG image
img = cv2.imread('ct_scan4.jpg', cv2.IMREAD_GRAYSCALE)

def normalize(image):
    lung_range = (-1000, 400)
    image = np.clip(image, *lung_range)
    image = (image - lung_range[0]) / (lung_range[1] - lung_range[0])
    image = (image * 255).astype(np.uint8) 
    return image

normalized = normalize(img)
equalized = cv2.equalizeHist(normalized)    # enhance the quality of the image
# give threshold to plot the lungs
_, thresholded = cv2.threshold(equalized, 190, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

# remove small obstacles
_, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
sizes = stats[:, -1]
max_label = 1 + np.argmax(sizes[1:])
cleaned = np.zeros_like(opened)
cleaned[labels == max_label] = 255

# Find contours 
contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
mask = np.zeros_like(img)   # create a black mask image to draw contour on it
right_lung = cv2.drawContours(mask, contours, 1, (255, 255, 255), cv2.FILLED)
left_lung = cv2.drawContours(right_lung, contours, 2, (255, 255, 255), cv2.FILLED)

# find contours of the mask to plot only the lungs
mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

lung_img = img.copy()   #create a copy of input image to plot the lung

# Draw contours of the lungs on the copy of the input image
cv2.drawContours(lung_img, mask_contours, -1, (0, 255, 0), 2)

# Display the images
cv2.imshow('Lung Contours', lung_img)
cv2.imshow('HU Image', normalized)
cv2.imshow('Lung Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
