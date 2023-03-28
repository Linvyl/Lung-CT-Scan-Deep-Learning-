import cv2

# Read the CT scan image
img = cv2.imread('ct_scan3.jpg', cv2.IMREAD_GRAYSCALE)

# Perform Otsu thresholding
threshold_value, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Perform morphological operations to clean up the mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find the convex hull of the mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
hull = []
for i in range(len(contours)):
    hull.append(cv2.convexHull(contours[i], False))
cv2.drawContours(mask, hull, -1, 255, -1)

# Dilate the mask
mask = cv2.dilate(mask, kernel, iterations=3)

# Display the resulting mask
cv2.imshow('Lung Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
