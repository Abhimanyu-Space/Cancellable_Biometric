import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_minutiae_from_image(image_path):
    """
    Reads a fingerprint, thins the ridges, and extracts ALL valid (X, Y) coordinates.
    Uses a dynamic Margin Crop to ignore the noisy borders of the scanner glass.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"CRITICAL ERROR: Could not find '{image_path}'.")

    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    try:
        skeleton = cv2.ximgproc.thinning(binary_img)
    except AttributeError:
        skeleton = np.zeros(binary_img.shape, np.uint8)
        eroded = np.zeros(binary_img.shape, np.uint8)
        temp = np.zeros(binary_img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        while not done:
            cv2.erode(binary_img, element, eroded)
            cv2.dilate(eroded, element, temp)
            cv2.subtract(binary_img, temp, temp)
            cv2.bitwise_or(skeleton, temp, skeleton)
            binary_img[:, :] = eroded[:, :]
            if cv2.countNonZero(binary_img) == 0:
                done = True

    # --- VISUALIZATION BLOCK FOR REPORT ---
    rows, cols = skeleton.shape
    margin_y = int(rows * 0.15)
    margin_x = int(cols * 0.15)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Skeletonized Input")
    plt.imshow(skeleton, cmap='gray')
    
    # Draw red rectangle to show the 15% Margin Crop area
    vis_img = cv2.cvtColor(skeleton.copy(), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis_img, (margin_x, margin_y), (cols-margin_x, rows-margin_y), (0, 0, 255), 2)
    
    plt.subplot(1, 2, 2)
    plt.title("15% Margin Crop (Safe Zone)")
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.show() # SAVE THIS AS margin_crop.png
    # --------------------------------------

    raw_points = []
    skeleton_norm = skeleton // 255 
    
    for i in range(margin_y, rows - margin_y):
        for j in range(margin_x, cols - margin_x):
            if skeleton_norm[i, j] == 1:
                p = [
                    skeleton_norm[i-1, j],   skeleton_norm[i-1, j+1], skeleton_norm[i, j+1],
                    skeleton_norm[i+1, j+1], skeleton_norm[i+1, j],   skeleton_norm[i+1, j-1],
                    skeleton_norm[i, j-1],   skeleton_norm[i-1, j-1], skeleton_norm[i-1, j]
                ]
                crossing_number = sum(1 for k in range(8) if p[k] == 0 and p[k+1] == 1)
                if crossing_number == 1 or crossing_number == 3:
                    raw_points.append([j, i]) 

    return np.array(raw_points, dtype=float)