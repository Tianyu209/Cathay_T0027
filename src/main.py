import cv2
import numpy as np
import json
import sys
import matplotlib.pyplot as plt
from Models import LuggageProcessor as lp


def main(image_path):
    model = lp.LuggageProcessor()
    model.depth_test(image_path=image_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        main(image_path)
    else:
        print("No image path provided.")
