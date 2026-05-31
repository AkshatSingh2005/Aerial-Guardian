from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

# Image
image_path = "data/visdrone/sequences/uav0000086_00000_v/0000001.jpg"

image = cv2.imread(image_path)

h, w = image.shape[:2]

# Baseline Detection
baseline_results = model(image, verbose=False)[0]

baseline_count = 0

for box in baseline_results.boxes:
    cls = int(box.cls[0])

    if cls == 0:  # person
        baseline_count += 1

print(f"Baseline Persons: {baseline_count}")

# -----------------------------
# Tiled Detection
# -----------------------------

tile_h = h // 2
tile_w = w // 2

tiles = [
    (0, 0, image[0:tile_h, 0:tile_w]),
    (tile_w, 0, image[0:tile_h, tile_w:w]),
    (0, tile_h, image[tile_h:h, 0:tile_w]),
    (tile_w, tile_h, image[tile_h:h, tile_w:w]),
]

all_boxes = []
all_scores = []

for tile_id, (offset_x, offset_y, tile) in enumerate(tiles):

    results = model(tile, verbose=False)[0]

    person_count = 0

    for box in results.boxes:

        cls = int(box.cls[0])

        if cls != 0:
            continue

        person_count += 1

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        confidence = float(box.conf[0])

        gx1 = int(x1 + offset_x)
        gy1 = int(y1 + offset_y)

        gx2 = int(x2 + offset_x)
        gy2 = int(y2 + offset_y)

        all_boxes.append([
            gx1,
            gy1,
            gx2 - gx1,
            gy2 - gy1
        ])

        all_scores.append(confidence)

    print(f"Tile {tile_id+1}: {person_count} persons")

print(f"\nRaw Tile Detections: {len(all_boxes)}")

# -----------------------------
# NMS
# -----------------------------

indices = cv2.dnn.NMSBoxes(
    all_boxes,
    all_scores,
    score_threshold=0.25,
    nms_threshold=0.5
)

final_count = len(indices)

print(f"Final Persons After NMS: {final_count}")

# -----------------------------
# Visualization
# -----------------------------

output = image.copy()

for idx in indices:

    i = int(idx)

    x, y, bw, bh = all_boxes[i]

    cv2.rectangle(
        output,
        (x, y),
        (x + bw, y + bh),
        (0, 255, 0),
        2
    )

cv2.imwrite("tiled_nms_output.jpg", output)

print("Saved: tiled_nms_output.jpg")