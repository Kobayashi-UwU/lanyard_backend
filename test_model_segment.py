import os
import cv2
import supervision as sv
from ultralytics import YOLO
from PIL import Image

model = YOLO('./model/segment/segment_best.pt')
image = Image.open("./test.png")
result = model.predict(image, conf=0.25)[0]

mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator(
    text_color=sv.Color.BLACK, text_position=sv.Position.CENTER)

detections = sv.Detections.from_ultralytics(result)

print(detections)
annotated_image = image.copy()
annotated_image = mask_annotator.annotate(
    annotated_image, detections=detections)
annotated_image = label_annotator.annotate(
    annotated_image, detections=detections)

sv.plot_image(annotated_image, size=(10, 10))
