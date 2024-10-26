from ultralytics import YOLO


model = YOLO("coco-turtle-100.pt")
print("Test")
print(model.names)


# class id 49 = sea turtle
results = model.predict(source="0", show=True, stream=True, classes=49)


