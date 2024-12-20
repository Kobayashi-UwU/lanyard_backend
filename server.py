from flask import Flask, request, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
from flask_cors import CORS
import supervision as sv

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load YOLO model
model = YOLO('./model/segment/segment_best.pt')


@app.route('/process-image', methods=['POST'])
def process_image():
    start_time = time.time()  # Start timing

    # Check if an image file is in the request
    if 'image' not in request.files:
        return {"error": "No image file provided"}, 400

    # Read the image file
    image_file = request.files['image']
    image_bytes = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    # Run the model on the image
    result = model.predict(image, conf=0.25)[0]

    # Extract detections
    detections = sv.Detections.from_ultralytics(result)

    # Perform additional processing on the detections segmentation
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK, text_position=sv.Position.CENTER)

    # Convert detections to a format suitable for Supervision
    annotated_image = image.copy()
    annotated_image = mask_annotator.annotate(
        annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(
        annotated_image, detections=detections)

    # Capture processing times
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing time: {processing_time:.2f} seconds")

    # Return the results as JSON
    return {
        "output": annotated_image,
        "processing_time": processing_time,
    }


@app.route('/blur-faces', methods=['POST'])
def blur_faces():
    # Check if both image and bounding boxes are provided
    if 'image' not in request.files or 'boxes' not in request.form:
        return {"error": "Image or bounding boxes missing"}, 400

    # Read the uploaded image
    file = request.files['image']
    image_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    # Parse the bounding boxes
    bounding_boxes = request.form.get('boxes')
    if not bounding_boxes:
        return {"error": "Bounding boxes missing"}, 400
    # Convert JSON string back to Python object
    bounding_boxes = eval(bounding_boxes)

    # Load a pre-trained face detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Tolerance percentage for horizontal range
    tolerance_percent = 0.2

    # Iterate over bounding boxes and check if they are below a face
    for box in bounding_boxes:
        lx1, ly1, lx2, ly2 = box['x1'], box['y1'], box['x2'], box['y2']

        # Check each detected face
        for (fx, fy, fw, fh) in faces:
            # Draw the face bounding box for debugging
            # cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh),
            #               (255, 0, 0), 2)  # Blue box for faces

            # Expand the horizontal range of the face
            fx1 = fx - int(fw * tolerance_percent)  # Left boundary
            fx2 = fx + fw + int(fw * tolerance_percent)  # Right boundary
            fy2 = fy + fh  # Bottom boundary (face ends here)

            # Draw the expanded range for debugging (optional)
            # Yellow box for expanded range
            # cv2.rectangle(image, (fx1, fy), (fx2, fy2), (0, 255, 255), 2)

            # Check if the lanyard is within the expanded face range
            if lx1 < fx2 and lx2 > fx1 and ly1 >= fy2:
                # Blur the face
                face_roi = image[fy:fy + fh, fx:fx + fw]
                face_roi = cv2.GaussianBlur(face_roi, (51, 51), 30)
                image[fy:fy + fh, fx:fx + fw] = face_roi

    # Save the image with annotations
    debug_image_path = './output_with_faces.png'
    cv2.imwrite(debug_image_path, image)

    # Save for debugging
    print(f"Debugging image saved at: {debug_image_path}")

    # Return the image with annotations
    _, buffer = cv2.imencode('.png', image)
    response_image = buffer.tobytes()

    return app.response_class(response_image, content_type='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5678)
