import json
import os

# Load COCO format
with open('/Users/Adrien/Documents/CPE/5ETI/WasteFlow/Assignment/Code/Yolov8_waste/Annotations/val/labels.json') as f:
    coco_data = json.load(f)

# YOLO format
output_dir = '/Users/Adrien/Documents/CPE/5ETI/WasteFlow/Assignment/Code/Yolov8_waste/dataset/labels/val'

for img in coco_data['images']:
    img_id = img['id']
    img_width = img['width']
    img_height = img['height']
    img_filename = os.path.splitext(img['file_name'])[0]
    
    # Annotations for the image
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
    
    # YOLO format annotation file
    with open(os.path.join(output_dir, f'{img_filename}.txt'), 'w') as yolo_f:
        for ann in annotations:
            category_id = ann['category_id'] - 1  # YOLO class ids starting at 0
            bbox = ann['bbox']
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            width = bbox[2] / img_width
            height = bbox[3] / img_height
            
            yolo_f.write(f'{category_id} {x_center} {y_center} {width} {height}\n')
