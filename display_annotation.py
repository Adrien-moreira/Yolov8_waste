import cv2
import matplotlib.pyplot as plt

# Path YOLO file of an annotated image
text_path = "/Users/Adrien/Documents/CPE/5ETI/WasteFlow/Assignment/Code/Yolov8_waste/dataset/labels/train/09_frame_033800.txt"
image_path = text_path.replace('labels', 'images').replace('.txt', '.png')  

# Load image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image.shape

# Read YOLO annotations 
annotations = []
with open(text_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width_b = float(parts[3])
        height_b = float(parts[4])
        annotations.append((class_id, x_center, y_center, width_b, height_b))

# Class names
class_names = {
    0: 'rigid_plastic',
    1: 'cardboard',
    2: 'metal',
    3: 'soft_plastic'
}

# Draw bounding boxes and class IDs
for ann in annotations:
    class_id, x_center, y_center, width_b, height_b = ann
    x = int((x_center - width_b / 2) * width)
    y = int((y_center - height_b / 2) * height)
    w = int(width_b * width)
    h = int(height_b * height)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red boxes
    cv2.putText(image, class_names[class_id], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

# Display the image
plt.imshow(image)
plt.axis('off')
plt.show()
