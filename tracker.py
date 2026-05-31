from ultralytics import YOLO
import supervision as sv
import cv2
import os

# Load model
model = YOLO("yolov8n.pt")

# ByteTrack
tracker = sv.ByteTrack()

sequence_path = "data/visdrone/sequences/uav0000086_00000_v"

frames = sorted(os.listdir(sequence_path))

# First frame
first_frame = cv2.imread(
    os.path.join(sequence_path, frames[0])
)

h, w = first_frame.shape[:2]

video_writer = cv2.VideoWriter(
    "tracking_output1.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    20,
    (w, h)
)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

from collections import defaultdict, deque

track_history = defaultdict(lambda: deque(maxlen=30))

for frame_name in frames:

    frame_path = os.path.join(sequence_path, frame_name)

    frame = cv2.imread(frame_path)

    results = model(frame, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(results)

    # keep only persons
    detections = detections[detections.class_id == 0]

    detections = tracker.update_with_detections(detections)

    for i in range(len(detections)):
    
        if detections.tracker_id[i] is None:
            continue

        tracker_id = int(detections.tracker_id[i])

        x1, y1, x2, y2 = detections.xyxy[i]

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        track_history[tracker_id].append(
            (center_x, center_y)
        )

    labels = [
        f"ID {tracker_id}"
        for tracker_id in detections.tracker_id
        if tracker_id is not None
    ]

    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )

    for tracker_id, points in track_history.items():

        for j in range(1, len(points)):

            cv2.line(
                annotated,
                points[j-1],
                points[j],
                (0,255,0),
                2
            )

    video_writer.write(annotated)

    print(frame_name)

video_writer.release()

print("Done")