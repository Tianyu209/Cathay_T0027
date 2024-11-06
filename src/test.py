import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from luggage_detection import yolov10_testing
from Depth import DepthEstimator as dm

# Load models
yolo = ""
depth_model = dm.DepthEstimator()
llm = " "

# Helper functions
def detect_luggage(image):
    results = yolo.detect(image)
    return results

def compute_depth(image_path=None, url=None):
    depth = depth_model.estimate_depth(image_path=image_path, url=url)
    return depth

def calculate_dimensions(bbox, depth):
    dimensions = llm.compute_dimensions(bbox, depth)
    return dimensions

def process_image(image_path):
    image = cv2.imread(image_path)
    luggage_bboxes = detect_luggage(image)
    results = []
    for bbox in luggage_bboxes:
        depth = compute_depth(image_path=image_path)
        dimensions = calculate_dimensions(bbox, depth)
        results.append({
            'bbox': bbox,
            'depth': depth,
            'dimensions': dimensions
        })
    return json.dumps(results, indent=4)

def show_depth_graph(depth):
    plt.imshow(depth)
    plt.colorbar()
    plt.title('Depth Map')
    plt.show()
def save_depth_output(depth, save_path):
    plt.imshow(depth)
    plt.savefig(save_path)
# Example usage
image_path = 'src\TestImages\OIP.jpeg'
depth_output = compute_depth(image_path=image_path)
save_path = f'src\ResultImages\depth_output_{image_path.split("\\")[-1]}.png'
save_depth_output(depth_output,save_path)
print(f"Depth output saved to {save_path}")
