import os
import cv2

# Directory path containing CT scan images
input_dir = 'F:/DeepLearning/Dataset/raw'

# Directory path where the output will be saved
output_dir = 'F:/DeepLearning/Dataset/processed'

# Kernel size used in the dilate function
kernel_size = (20, 20)

# Get a list of image files in the input directory
image_files = os.listdir(input_dir)

# Loop through each image file and process it
for image_file in image_files:
    # Read the image and convert to grayscale
    img = cv2.imread(os.path.join(input_dir, image_file))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding method to extract lung mask
    ret, lung_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up the lung mask
    lung_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    lung_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))

    # Find the convex hull of the lung mask
    lung_contours, _ = cv2.findContours(lung_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lung_hull = cv2.convexHull(lung_contours[0])

    # Dilate the lung mask
    lung_mask = cv2.dilate(lung_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size))

    # Save the result to the output directory
    output_file = os.path.join(output_dir, 'lung_mask_' + image_file)
    cv2.imwrite(output_file, lung_mask)