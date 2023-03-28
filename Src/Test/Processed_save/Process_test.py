import os
import cv2
import numpy as np

input_dir = 'F:/DeepLearning/Dataset/Normal cases/Raw'
output_dir = 'F:/DeepLearning/Dataset/Normal cases/Processed_Test'

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

    # Enhance the quality of the image
    equalized = cv2.equalizeHist(normalized)

    # Threshold the image to plot the lungs
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
    mask = np.zeros_like(img)   # create a black mask image to draw contour on it
    right_lung = cv2.drawContours(mask, contours, 1, (255, 255, 255), cv2.FILLED)
    left_lung = cv2.drawContours(right_lung, contours, 2, (255, 255, 255), cv2.FILLED)

    # find contours of the mask to plot only the lungs
    mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lung_img = img.copy()   #create a copy of input image to plot the lung

    # Draw contours of the lungs on the copy of the input image
    cv2.drawContours(lung_img, mask_contours, -1, (0, 255, 0), 2)

    # Save the output mask
    cv2.imwrite(output_path, cleaned)

    print(f"Processed {input_path} and saved mask to {output_path}")


# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through all files in the input directory
for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # Process and save the image
    process_and_save(input_path, output_path)
