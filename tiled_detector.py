from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

image_path = "data/visdrone/sequences/uav0000086_00000_v/0000001.jpg"

frame = cv2.imread(image_path)

h, w = frame.shape[:2]

baseline = model(frame, verbose=False)[0]

baseline_count = 0

for box in baseline.boxes:
    if int(box.cls) == 0:
        baseline_count += 1

print("Baseline:", baseline_count)
print()

# split into 4 tiles

tile1 = frame[0:h//2, 0:w//2]
tile2 = frame[0:h//2, w//2:w]
tile3 = frame[h//2:h, 0:w//2]
tile4 = frame[h//2:h, w//2:w]

tiles = [tile1, tile2, tile3, tile4]

for i, tile in enumerate(tiles):

    results = model(tile, verbose=False)[0]

    person_count = 0

    for box in results.boxes:

        if int(box.cls) == 0:
            person_count += 1

    print(f"Tile {i+1}: {person_count} persons")