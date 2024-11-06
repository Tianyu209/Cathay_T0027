import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

# from LuggageDetect import yolov10_testing
# from Models.Depth import DepthEstimator as dm
# from Models.Dimension import DimensionCal as dc

from LuggageDetect import yolov10_testing
from Depth import DepthEstimator as dm
from Dimension import DimensionCal as dc
class LuggageProcessor:
    
    def __init__(self):
        self.yolo = None  # Replace with yolo model
        self.depth_model = dm.DepthEstimator()
        self.llm = None  # Replace with llm model
    #helper functions
    def detect_luggage(self, image):
        return self.yolo.detect(image)
    
    def compute_depth(self, image_path=None, url=None):
        return self.depth_model.estimate_depth(image_path=image_path, url=url)
    
    def calculate_dimensions(self, bbox, depth):
        return self.llm.compute_dimensions(bbox, depth)
    
    def process_image(self, image_path):
        image = cv2.imread(image_path)
        luggage_bboxes = self.detect_luggage(image)
        results = []
        for bbox in luggage_bboxes:
            depth = self.compute_depth(image_path=image_path)
            dimensions = self.calculate_dimensions(bbox, depth)
            results.append({
                'bbox': bbox,
                'depth': depth,
                'dimensions': dimensions
            })
        return json.dumps(results, indent=4)
    
    def save_depth_output(self, depth, save_path):
        plt.imshow(depth)
        plt.axis('off') 
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Depth output saved to {save_path}")

    def show_depth_graph(self, depth):
        plt.imshow(depth)
        plt.colorbar()
        plt.title('Depth Map')
        plt.show()

    def depth_test(self,image_path):
        depth_output = self.compute_depth(image_path=image_path)
        save_path = f'Models\ResultImages\depth_output_{image_path.split("\\")[-1]}.png'
        self.save_depth_output(depth_output, save_path)
model = LuggageProcessor()
path = r"TestImages\OIP.jpeg"
model.depth_test(image_path = path)