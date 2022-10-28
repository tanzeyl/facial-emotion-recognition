from PIL import Image
import os
from boundingBoxUtil import Rect
import sys
import numpy as np

def process_data(emotion_raw, mode):
        size = len(emotion_raw)
        emotion_unknown = [0.0] * size
        emotion_unknown[-2] = 1.0
        sum_list = sum(emotion_raw)
        emotion = [0.0] * size
        if mode == 'majority':
            maxval = max(emotion_raw)
            if maxval > 0.5 * sum_list:
                emotion[np.argmax(emotion_raw)] = maxval
            else:
                emotion = emotion_unknown
        return [float(i) / sum(emotion) for i in emotion]

row = ['ck+792.png', '(0, 0, 48, 48)', '0', '0', '0', '0', '0', '0', '1', '0', '0']
image_path = os.path.join("./data\CK+Train", row[0])
image_data = Image.open(image_path)
image_data.load()
box = list(map(int, row[1][1:-1].split(',')))
print(box)
face_rc = Rect(box)
emotion_raw = list(map(float, row[2:len(row)]))
print(f"Before processing: {emotion_raw}")
emotion = process_data(emotion_raw, "majority")
print(f"After processing: {emotion_raw}")
idx = np.argmax(emotion)
print(f"Index of max: {idx}")
