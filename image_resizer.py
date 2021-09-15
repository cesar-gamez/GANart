#This file is to resize every image to 128x128 and convert to an array for image processing and training

import os
import numpy as np
from PIL import Image

IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
IMAGE_DIR = '/Image/Dir' #add path with test data

training_data = []
images_path = IMAGE_DIR

for filename in os.listdir(images_path):
    path = os.path.join(images_path, filename)
    image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
    training_data.append(np.asarray(image))
    print(f"Resizing {filename}")

training_data = np.reshape(training_data ,(-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
training_data = training_data/127.5-1

np.save('monet_data.npy', training_data)
print('Images Fully Processed and Saved')
