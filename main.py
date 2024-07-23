import time
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.yaml") 

start_time = time.time()

# Train the model with quick testing settings
results = model.train(
    data="/Users/Adrien/Documents/CPE/5ETI/WasteFlow/Assignment/Code/Yolov8_waste/config.yaml", 
    epochs=1,
    imgsz=320,  # reduce image size
    batch=4,  
    project="/Users/Adrien/Documents/CPE/5ETI/WasteFlow/Assignment/Code/Yolov8_waste/runs",
    name="custom_experiment"
)

end_time = time.time()

# Duration of the training
training_duration = end_time - start_time
print(f"Training took {training_duration / 60:.2f} minutes")
