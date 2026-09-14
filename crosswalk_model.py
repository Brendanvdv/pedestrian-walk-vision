from ultralytics import YOLO

#load model
model = YOLO("yolov8n.pt")

#train model
results = model.train(data=r"C:\Users\Brend\Downloads\crosswalk_instance\data.yaml", epochs=100, imgsz=640)