from transformers import pipeline
from PIL import Image
import requests
#A function class for depth estimation using Depth-Anything-V2-Small model
class DepthEstimator:
    def __init__(self):
        # Load the model
        self.pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

    def estimate_depth(self, image_path=None, url=None, retur_image=False):
        # Load image from URL or file path
        if url:
            image = Image.open(requests.get(url, stream=True).raw)
        elif image_path:
            image = Image.open(image_path)
        else:
            raise ValueError("Provide either image_path or url.")
        
        # Run depth estimation and return a depth graph
        depth = self.pipe(image)["depth"]
        if retur_image:
            return depth, image
        return depth
