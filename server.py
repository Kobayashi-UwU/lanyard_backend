from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import time
from flask_cors import CORS
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load YOLO model
model = YOLO('./model/segment/segment_best2.pt')


def display_image(image, detections):
    mask_annotator = sv.MaskAnnotator()

    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK, text_position=sv.Position.CENTER)

    annotated_image = image.copy()
    annotated_image = mask_annotator.annotate(
        annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(
        annotated_image, detections=detections)

    return annotated_image


@app.route('/process-image', methods=['POST'])
def process_image():
    start_time = time.time()  # Start timing

    # Check if an image file is in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Read the image file
    image_file = request.files['image']
    image_bytes = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    # Make predictions
    result = model.predict(image, conf=0.25)[0]
    detections = sv.Detections.from_ultralytics(result)

    segmented_found = []

    for mask in detections.mask:
        mask = np.array(mask)  # Ensure mask is a numpy array

        # Ensure the mask is binary
        binary_mask = (mask > 0).astype(np.uint8)

        # Apply the mask to the original image
        segmented_lanyard = cv2.bitwise_and(image, image, mask=binary_mask)

        # Get the pixels where the mask is applied
        pixels = segmented_lanyard[binary_mask == 1]

        # Calculate the median color for each channel
        median_color_bgr = np.median(pixels, axis=0).astype(int)

        # Convert BGR to RGB
        median_color_rgb = [int(median_color_bgr[2]), int(
            median_color_bgr[1]), int(median_color_bgr[0])]

        # Find the bounding box of the mask for location
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])

        # Convert to a list of (x, y) points
        points = contours[0].reshape(-1, 2).tolist()

        segmented_found.append({
            "location": [int(x), int(y), int(w), int(h)],
            "color": median_color_rgb,  # Store in RGB format
            "segmentation_points": points  # Store all segmentation points
        })

    end_time = time.time()  # End timing
    processing_time = end_time - start_time

    segmented_image = display_image(image, detections)
    # Convert the annotated image to base64 for easier transport (optional)
    _, buffer = cv2.imencode('.jpg', segmented_image)
    encoded_image = buffer.tobytes()

    # Return the results as JSON
    return jsonify({
        "segmented_image": encoded_image.hex(),  # Use hex for base64-like transport
        "segmented_found": segmented_found,
        "processing_time": processing_time,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5678)
