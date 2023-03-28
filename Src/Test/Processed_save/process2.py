import os
import cv2
import numpy as np

# Set the input and output directories
input_dir = 'F:/DeepLearning/Dataset/Normal cases/Raw'
output_dir = 'F:/DeepLearning/Dataset/Normal cases/Processed2'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def normalize(image):
    lung_range = (-1000, 400)
    image = np.clip(image, *lung_range)
    image = (image - lung_range[0]) / (lung_range[1] - lung_range[0])
    image = (image * 255).astype(np.uint8) 
    return image

# Loop over all files in the input directory
for filename in os.listdir(input_dir):
    # Check if the file is an image
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Normalize the image
        normalized = normalize(img)

        # Enhance the quality of the image
        equalized = cv2.equalizeHist(normalized)

        # Threshold to plot the lungs
        _, thresholded = cv2.threshold(equalized, 190, 255, cv2.THRESH_BINARY)

        # Morphological closing and opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        # Remove small obstacles
        _, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
        sizes = stats[:, -1]
        max_label = 1 + np.argmax(sizes[1:])
        cleaned = np.zeros_like(opened)
        cleaned[labels == max_label] = 255

        # Find contours 
        contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Create a black mask image to draw contour on it
        mask = np.zeros_like(img)

        # Draw contours of the lungs on the mask image
        mask_contours = cv2.drawContours(mask, contours, -1, (255, 255, 255), cv2.FILLED)

        # Find contours of the mask to plot only the lungs
        lung_contours, _ = cv2.findContours(mask_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the input image to plot the lung
        lung_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # Draw contours of the lungs on the copy of the input image
        cv2.drawContours(lung_img, lung_contours, -1, (0, 255, 0), 2)

        # Save the output image with the same filename as the input image
        output_path = os.path.join(output_dir, filename.replace(".jpg", "") + "_mask.jpg")
        cv2.imwrite(output_path, lung_img)
