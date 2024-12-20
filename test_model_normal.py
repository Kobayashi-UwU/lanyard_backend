import cv2
from ultralytics import YOLO
import time

# Load YOLO model
model = YOLO('./model/v3/best.pt')

# Load test image
test_image_path = './test.png'
image = cv2.imread(test_image_path)

start_time = time.time()  # Start timing

# Perform inference
results = model(test_image_path)

# Loop through detections and draw them
for result in results:
    for box in result.boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = box.conf[0].item()  # Confidence score
        class_id = int(box.cls[0].item())  # Class ID
        label = f"{model.names[class_id]} {confidence:.2f}"

        # Draw rectangle and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

end_time = time.time()  # End timing
processing_time = end_time - start_time
print(f"Processing time: {processing_time:.2f} seconds")

# Save or display the resulting image
output_path = './output.png'
cv2.imwrite(output_path, image)

print(f"Output saved to {output_path}")
