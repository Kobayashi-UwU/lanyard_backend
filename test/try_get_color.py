import cv2
import numpy as np
from sklearn.cluster import KMeans

# Function to detect lines using Hough Transform


def detect_lanyard_lines(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi /
                            180, threshold=50, minLineLength=30, maxLineGap=10)
    return lines, edges

# Function to filter vertical/diagonal lines


def filter_lanyard_lines(lines, slope_threshold=0.5):
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = abs((y2 - y1) / (x2 - x1 + 1e-6))  # Avoid division by zero
            if slope > slope_threshold:  # Focus on diagonal/vertical lines
                filtered_lines.append((x1, y1, x2, y2))
    return filtered_lines

# Function to extract region between lines


def extract_region_between_lines(image, lines, thickness=10):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for x1, y1, x2, y2 in lines:
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
    lanyard_region = cv2.bitwise_and(image, image, mask=mask)
    return lanyard_region, mask

# Function to get the dominant color using K-means


def get_dominant_color(image, k=3):
    pixels = image.reshape(-1, 3)
    # Exclude black pixels (mask background)
    pixels = pixels[np.any(pixels > 0, axis=1)]
    if len(pixels) == 0:
        return (0, 0, 0)  # Return black if no pixels found
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]
    return tuple(map(int, dominant))

# Main function


def analyze_lanyard_color(image_path):
    # Load the cropped YOLO output image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not loaded.")
        return

    # Detect lines in the image
    lines, edges = detect_lanyard_lines(image)
    if lines is None:
        print("No lines detected.")
        return

    # Filter vertical/diagonal lines
    filtered_lines = filter_lanyard_lines(lines)
    if not filtered_lines:
        print("No suitable lanyard lines found.")
        return

    # Visualize detected lines
    line_image = image.copy()
    for x1, y1, x2, y2 in filtered_lines:
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Extract the region between the detected lines
    lanyard_region, mask = extract_region_between_lines(image, filtered_lines)

    # Get the dominant color
    dominant_color = get_dominant_color(lanyard_region)
    print(f"Dominant Lanyard Color (BGR): {dominant_color}")

    # Display the results
    cv2.imshow("Original Image", image)
    cv2.imshow("Edges Detected", edges)
    cv2.imshow("Lines Detected", line_image)
    cv2.imshow("Lanyard Region", lanyard_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Run the code
# Replace with your cropped YOLO image path
image_path = "yolo_output3.png"
analyze_lanyard_color(image_path)
