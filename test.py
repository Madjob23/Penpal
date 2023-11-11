import tensorflow as tf
import cv2 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

model = tf.keras.models.load_model('penpal.model')
image_number=1
while os.path.isfile(f"Digits/Digit_{image_number}.png"):
    try:
        img = cv2.imread(f"Digits/Digit_{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This Digit According to me is:{np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        
    except:
        print("Error!")
    finally:
        image_number+=1