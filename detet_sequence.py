from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

sequence_path = "data/visdrone/sequences/uav0000086_00000_v"

frames = sorted(os.listdir(sequence_path))

first_frame = cv2.imread(
    os.path.join(sequence_path, frames[0])
)

h, w = first_frame.shape[:2]

out = cv2.VideoWriter(
    "baseline_detection.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    20,
    (w, h)
)

for frame_name in frames:

    frame_path = os.path.join(sequence_path, frame_name)

    results = model(frame_path)

    annotated = results[0].plot()

    out.write(annotated)

    print("Processed:", frame_name)

out.release()

print("Done")