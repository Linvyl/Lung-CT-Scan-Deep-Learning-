import os
import cv2
import numpy as np

# Set the path of the input and output directories
input_dir = 'F:/DeepLearning/Dataset/Normal cases/Raw'
output_dir = 'F:/DeepLearning/Dataset/Normal cases/Processed1'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def normalize(image):
    lung_range = (-1000, 400)
    image = np.clip(image, *lung_range)
    image = (image - lung_range[0]) / (lung_range[1] - lung_range[0])
    image = (image * 255).astype(np.uint8) 
    return image

for filename in os.listdir(input_dir):
    if not filename.endswith('.jpg'):
        continue

    # Load the JPG image
    img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)

    normalized = normalize(img)
    equalized = cv2.equalizeHist(normalized)
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
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), cv2.FILLED)

    # Save the mask with the same filename as the original image
    output_filename = os.path.splitext(filename)[0] + '.jpg'
    cv2.imwrite(os.path.join(output_dir, output_filename), mask)

    print(f"Processed {filename} and saved mask as {output_filename}")

print("Done")
