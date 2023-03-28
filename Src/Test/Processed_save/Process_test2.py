import os
import cv2
import numpy as np

def normalize(image):
    lung_range = (-1000, 400)
    image = np.clip(image, *lung_range)
    image = (image - lung_range[0]) / (lung_range[1] - lung_range[0])
    image = (image * 255).astype(np.uint8) 
    return image

def process_and_save(input_path, output_path):
    # Load the image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    # Normalize the image
    normalized = normalize(img)
    
    # Equalize the image
    equalized = cv2.equalizeHist(normalized)
    
    # Apply threshold to find the lungs
    _, thresholded = cv2.threshold(equalized, 190, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the image
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
    
    # Create a mask to draw the contours on
    mask = np.zeros_like(img)
    
    # Draw contours for the right lung
    right_lung_mask = cv2.drawContours(mask, contours, 1, (255, 255, 255), cv2.FILLED)
    
    # Create a new mask for the left lung
    left_lung_mask = np.zeros_like(img)
    
    # Draw contours for the left lung
    left_lung_mask = cv2.drawContours(left_lung_mask, contours, 2, (255, 255, 255), cv2.FILLED)
    
    # Save the mask for the left lung
    filename = os.path.splitext(os.path.basename(input_path))[0]
    output_file = os.path.join(output_path, filename + '_mask.jpg')
    cv2.imwrite(output_file, left_lung_mask)
    
    # Find contours of the mask to plot only the lungs
    mask_contours, _ = cv2.findContours(left_lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the input image to plot the lungs on
    lung_img = img.copy()
    
    # Draw contours of the lungs on the copy of the input image
    cv2.drawContours(lung_img, mask_contours, -1, (0, 255, 0), 2)
    
    # Save the image with the lung contours
    output_file = os.path.join(output_path, filename + '.jpg')
    cv2.imwrite(output_file, lung_img)

# Process all images in the input directory
input_dir = 'F:/DeepLearning/Dataset/Normal cases/Raw'
output_dir = 'F:/DeepLearning/Dataset/Normal cases/Processed_Test2'

os.makedirs
