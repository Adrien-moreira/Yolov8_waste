from ultralytics import YOLO

# Load the trained model
model = YOLO("/Users/Adrien/Documents/CPE/5ETI/WasteFlow/Assignment/Code/Yolov8_waste/runs/custom_experiment3/weights/best.pt")

# Run inference on an unseen test image
results = model.predict(source="/Users/Adrien/Documents/CPE/5ETI/WasteFlow/Assignment/Code/dataset_old/test/data/08_frame_004600.PNG", save=True)

# Print or analyze the results
print("Predictions:", results)
