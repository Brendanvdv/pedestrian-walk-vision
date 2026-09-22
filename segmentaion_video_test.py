from ultralytics import YOLO

model = YOLO(r"model_weights\segmentation_dataset\best.pt")

video_path = "wts_sample.mp4"

results = model(
    source=video_path,
    save=True,
    project=r"C:\Users\Brend\Documents\pedestrian-walk-vision\outputs",
    name="segmentation_test",
    exist_ok=True,
    conf=0.2,
    stream=True,
)

for r in results:
    if r.masks is not None:
        print(f"Frame {r.path} | {len(r.masks.xy)} crosswalk(s) detected")
        for poly in r.masks.xy:
            print(poly)
    else:
        print(f"Frame {r.path} | no crosswalk detected")