from deepforest import main
from deepforest import get_data
import os
import matplotlib.pyplot as plt
import cv2
model = main.deepforest()
model.use_release()

img = model.predict_tile("doc/trees_rgb.jpg", return_plot = True,
        patch_size=500,patch_overlap=0.1)

plt.imshow(img[:,:,::-1])
plt.show()
