from ultralytics import YOLO
from PIL import Image
import numpy as np
import supervision as sv
import cv2

# Load the YOLO model
model = YOLO('segment_best.pt')

# Open the image from a local file path
image = Image.open('./test_group.jpg')

# Make predictions
result = model.predict(image, conf=0.25)[0]

# Convert results to Supervision Detections
detections = sv.Detections.from_ultralytics(result)

# Process each detected lanyard
for i, mask in enumerate(detections.mask):
    mask = np.array(mask)  # Convert to numpy array if not already

    # Ensure the mask is binary
    mask = (mask > 0).astype(np.uint8)

    # Apply the mask to the original image
    image_array = np.array(image)
    segmented_lanyard = cv2.bitwise_and(image_array, image_array, mask=mask)

    # Convert to RGB (if necessary)
    if segmented_lanyard.shape[-1] == 4:  # Check if it's RGBA
        segmented_lanyard = cv2.cvtColor(segmented_lanyard, cv2.COLOR_RGBA2RGB)

    # Get the pixels where the mask is applied
    pixels = segmented_lanyard[mask == 1]

    # Calculate the median color for each channel
    median_color = np.median(pixels, axis=0)

    # Convert the median color to an integer RGB tuple
    median_color = tuple(median_color.astype(int))

    # Print or use the median color
    print(f"Lanyard {i + 1} median color: {median_color}")

    # Visualize the segmented area and median color
    segmented_lanyard = cv2.cvtColor(segmented_lanyard, cv2.COLOR_RGB2BGR)
    cv2.imshow(f'Segmented Lanyard {i + 1}', segmented_lanyard)
    cv2.waitKey(0)

cv2.destroyAllWindows()
