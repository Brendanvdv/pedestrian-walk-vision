from ultralytics import YOLO

model = YOLO(r"C:\Users\Brend\Downloads\best.pt")

results = model(r"C:\Users\Brend\Downloads\testy.png", conf=0.2)  # conf = confidence threshold

for r in results:
    r.show()          # pops up the image with boxes drawn
    r.save()          # saves it to disk instead
    # print(r.boxes)    # raw data: box coordinates, confidence, class
    print(r.masks.xy)   # the actual polygon points, per detected crosswalk