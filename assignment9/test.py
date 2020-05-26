import sys
import os
import numpy as np
from keras.models import load_model
from PIL import Image
import PIL
import matplotlib.pyplot as plt

model_file = sys.argv[1]


currDir = os.getcwd()


imagePath = os.path.join(currDir, sys.argv[2])
print(imagePath)

if not os.path.isdir(imagePath):
    os.mkdir(imagePath)

examples=100
dim=(10,10)
figsize=(10,10)
model = load_model(model_file)
noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
generated_images = model.predict(noise)
generated_images = generated_images.reshape(100,28,28)
plt.figure(figsize=figsize)
for i in range(generated_images.shape[0]):
    plt.subplot(dim[0], dim[1], i+1)
    plt.imshow(generated_images[i], interpolation='nearest')
    plt.axis('off')
plt.tight_layout()
savePath = os.path.join(imagePath, "image"+'.JPEG')
plt.savefig(savePath)