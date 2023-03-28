import cv2
import numpy as np
import os

def process_image(img_path):
    # Load the image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Normalize the image
    def normalize(image):
        lung_range = (-1000, 400)
        image = np.clip(image, *lung_range)
        image = (image - lung_range[0]) / (lung_range[1] - lung_range[0])
        image = (image * 255).astype(np.uint8)
        return image
    normalized = normalize(img)

    # Equalize the image
    equalized = cv2.equalizeHist(normalized)

    # Threshold the image
    _, thresholded = cv2.threshold(equalized, 190, 255, cv2.THRESH_BINARY)

    # Morphological operations
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

    # Draw contours of the lungs on the copy of the input image
    mask = np.zeros_like(img)
    right_lung = cv2.drawContours(mask, contours, 1, (255, 255, 255), cv2.FILLED)
    left_lung = cv2.drawContours(right_lung, contours, 2, (255, 255, 255), cv2.FILLED)
    mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lung_img = img.copy()
    cv2.drawContours(lung_img, mask_contours, -1, (0, 255, 0), 2)

    return mask, lung_img

# Set input and output folder paths
input_folder = "F:/DeepLearning/Dataset/Normal cases/Raw"
output_folder = "F:/DeepLearning/Dataset/Normal cases/Processed3"

# Create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder and save the results to the output folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Process the image
        img_path = os.path.join(input_folder, filename)
        mask, lung_img = process_image(img_path)

        # Save the mask to the output folder
        mask_path = os.path.join(output_folder, f"{filename}_mask.jpg")
        cv2.imwrite(mask_path, mask)

        # # Save the lung image to the output folder
        # lung_path = os.path.join(output_folder, f"{filename}_lung.jpg")
        # cv2.imwrite(lung_path, lung_img)

    else:
        continue
