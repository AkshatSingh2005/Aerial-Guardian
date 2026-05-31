from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

image_path = "data/visdrone/sequences/uav0000086_00000_v/0000001.jpg"

results = model(image_path)

annotated = results[0].plot()

cv2.imshow("Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()