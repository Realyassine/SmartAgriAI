import cv2   #correct it
import numpy as np
import tensorflow as tf

# Load model and labels
model = tf.keras.applications.MobileNetV2(weights='imagenet')
labels_path = tf.keras.utils.get_file('imagenet_class_index.json',
                                      'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json')

import json
with open(labels_path) as f:
    labels = json.load(f)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)
    

def predict_disease(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_name = labels[str(predicted_class)][1]
    print(f"Prediction: {class_name}")

image_path = 'images/download.jpg'
predict_disease(image_path)
