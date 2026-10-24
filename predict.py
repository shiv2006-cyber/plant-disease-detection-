"""Classify a plant leaf image.

Usage: python predict.py path/to/leaf.jpg
"""
import json
import sys

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image as kimage

IMG_SIZE = 224
TOP_K = 3

if len(sys.argv) != 2:
    sys.exit("Usage: python predict.py path/to/leaf.jpg")

model = keras.models.load_model("plant_disease_model.keras")
with open("class_names.json") as f:
    class_names = json.load(f)

img = kimage.load_img(sys.argv[1], target_size=(IMG_SIZE, IMG_SIZE))
batch = np.expand_dims(kimage.img_to_array(img), 0)  # model rescales internally
probs = model.predict(batch, verbose=0)[0]

for i in np.argsort(probs)[::-1][:TOP_K]:
    print(f"{class_names[i]:50s} {probs[i] * 100:5.1f}%")
