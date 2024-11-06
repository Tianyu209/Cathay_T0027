import cv2
import numpy as np
import json

import matplotlib.pyplot as plt
from Models import LuggageProcessor as lp
model = lp.LuggageProcessor()
path = 'Models\TestImages\dp.jpg'

model.depth_test(image_path = path)