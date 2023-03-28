# Define the input and output directories
# input_dir = 'F:/DeepLearning/Dataset/Normal cases/Raw'
# output_dir = 'F:/DeepLearning/Dataset/Normal cases/Processed'

import cv2
import numpy as np
import os

# Define the input and output directories
input_dir = 'F:/DeepLearning/Dataset/Normal cases/Raw'
output_dir = 'F:/DeepLearning/Dataset/Normal cases/Processed'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def normalize(image):
    lung_range = (-1000, 400)
    image = np.clip(image, *lung_range)
    image = (image - lung_range[0]) / (lung_range[1] - lung_range[0])
    image = (image * 255).astype(np.uint8) 
    return image

# Loop over all the image files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg'):
        
        # Load the image
        img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)
        
        # Normalize the image
        normalized = normalize(img)
        
        # Equalize the image
        equalized = cv2.equalizeHist(normalized)
        
        # Threshold the image
        _, thresholded = cv2.threshold(equalized, 190, 255, cv2.THRESH_BINARY)
        
        # Apply morphology operations to close and open the image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        # Remove small obstacles
        _, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
        sizes = stats[:, -1]
        max_label = 1 + np.argmax(sizes[1:])
        cleaned = np.zeros_like(opened)
        cleaned[labels == max_label] = 255

        # Find contours of the lungs
        contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(img)
        mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), cv2.FILLED)

        # Write the mask image to the output directory
        output_filename = os.path.join(output_dir, os.path.splitext(filename)[0] + '_mask.jpg')
        cv2.imwrite(output_filename, mask)

        print(f'{filename} processed and saved to {output_filename}')

print('All images processed and masks saved.')

